# limit the number of cpus used by high performance libraries
import sys

sys.path.insert(0, './blazeface')

import argparse
import copy
import os
from pathlib import Path
import numpy as np
import cv2

from blazeface.utils.datasets import LoadImages, LoadStreams, IMG_FORMATS, VID_FORMATS
from blazeface.utils.general import (check_img_size, non_max_suppression_face, scale_coords, scale_coords_landmarks,
                                     check_imshow, xyxy2xywh, increment_path)
from blazeface.utils.onnx_utils import select_device, load_onnx_model
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # blazeface deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def show_results(img, xyxy, conf, landmarks):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def detect(opt):
    source, blazeface_model, deep_sort_model, show_vid, save_vid, imgsz, half, project, exist_ok, max_stride = (
        opt.source, opt.blazeface_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, opt.imgsz,
        opt.half, opt.project, opt.exist_ok, opt.max_stride)

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    imgsz = check_img_size(imgsz, s=max_stride)  # check image size

    # Load blazeface model
    model = load_onnx_model(blazeface_model, select_device(opt.device))

    # Half
    half &= opt.device.lower() != 'cpu'  # half precision only supported on CUDA

    show_vid &= check_imshow()

    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz, stride=max_stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=max_stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                'cpu',
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = ['face']

    # Run tracking
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        im = im.astype('float32')
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        im = im[::-1, ...] # RGB to BGR
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model.run([model.get_outputs()[0].name], {model.get_inputs()[0].name: im})[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(Path(save_dir) / p.name)  # im.jpg

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                det[:, 5:15] = scale_coords_landmarks(im.shape[2:], det[:, 5:15], im0.shape).round()

                xywhs = xyxy2xywh(det[:, :4])
                confs = det[:, 4]
                clss = det[:, 15]

                im0_display = im0.copy()
                for j in range(det.shape[0]):
                    xyxy = det[j, :4].tolist()
                    conf = det[j, 4]
                    landmarks = det[j, 5:15].tolist()
                    im0_display = show_results(im0_display, xyxy, conf, landmarks)

                # pass detections to deepsort
                outputs[i] = deepsort_list[i].update(xywhs, confs, clss, im0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blazeface_model', type=str, default='onnx_models/blazeface.onnx',
                        help='blazeface_model.onnx path(s)')
    parser.add_argument('--max-stride', type=int, default=16, help='blazeface model max stride')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='face inference conf threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='face IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. cpu or gpu')
    parser.add_argument('--show-vid', action='store_true', help='display rpph video results')
    parser.add_argument('--save-vid', action='store_true', help='save video rppg results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    detect(opt)
