# limit the number of cpus used by high performance libraries
import sys

sys.path.insert(0, './blazeface')

import argparse
import os
from pathlib import Path
import cv2
import torch

from blazeface.utils.datasets import LoadImages, LoadStreams, IMG_FORMATS, VID_FORMATS
from blazeface.utils.general import check_img_size, non_max_suppression_face, scale_coords, check_imshow, xyxy2xywh, \
    increment_path
from blazeface.utils.torch_utils import select_device, load_model
from blazeface.utils.plots import Annotator
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # blazeface deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    is_file = Path(opt.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = opt.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or (is_url and not is_file)

    imgsz = check_img_size(opt.imgsz, s=opt.max_stride)  # check image size

    face_crop_height, face_crop_width = opt.face_crop_size

    device = select_device(opt.device)

    # Load face model
    face_model = load_model(weights=opt.blazeface_model, device=device)

    # Half
    opt.half &= opt.device.lower() != 'cpu'  # half precision only supported on CUDA

    opt.show_vid &= check_imshow()

    exp_name = opt.blazeface_model.split('/')[-1].split('.')[0] + "_" + opt.deep_sort_model.split('/')[-1].split('.')[
        0]
    save_dir = increment_path(Path(opt.project) / exp_name,
                              exist_ok=opt.exist_ok)  # increment run if project name exists
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Dataloader
    if webcam:
        print('loading streams:', opt.source)
        dataset = LoadStreams(opt.source, img_size=imgsz, stride=opt.max_stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(opt.source, img_size=imgsz, stride=opt.max_stride)
        nr_sources = 1
    vid_path, vid_writer = [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                opt.deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                width=face_crop_width, height=face_crop_height,
                use_larger_box=opt.use_larger_box,
                larger_box_coef=opt.larger_box_coef
            )
        )
    outputs = [None] * nr_sources

    # Run tracking
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        im = torch.from_numpy(im).to(device).float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[[2, 1, 0], ...]  # RGB to BGR
            im = im[None]  # expand for batch dim
        else:
            im = im[:, [2, 1, 0], ...]

        # Inference
        pred = face_model(im)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if opt.source.endswith(tuple(VID_FORMATS)):
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            annotator = Annotator(im0, line_width=2)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # det[:, 5:15] = scale_coords_landmarks(im.shape[2:], det[:, 5:15], im0.shape).round()

                xywhs = xyxy2xywh(det[:, :4])
                confs = det[:, 4]
                clss = det[:, 15].cpu()

                # pass detections to deepsort
                outputs[i] = deepsort_list[i].update(xywhs, confs, clss, im0)

                if len(outputs[i]) > 0:
                    for output in outputs[i]:
                        [x1, y1, x2, y2, id, _], conf, face_image, face_crop_image = output
                        label = f'id:{id}  conf:{conf:.2f}'

                        if opt.save_img:
                            # face_img_save_dir = Path(save_path) / 'face_images'/ f'track_{id}'
                            crop_face_img_save_dir = Path(save_path) / 'crop_face_images'/ f'track_{id}'
                            # face_img_save_dir.mkdir(parents=True, exist_ok=True)
                            crop_face_img_save_dir.mkdir(parents=True, exist_ok=True)
                            # cv2.imwrite(str(face_img_save_dir / f'{frame_idx}.png'), face_image)
                            cv2.imwrite(str(crop_face_img_save_dir / f'{frame_idx}.png'), face_crop_image)

                        if opt.save_vid or opt.show_vid:  # Add bbox to image
                            annotator.box_label([x1, y1, x2, y2], label)


            else:
                deepsort_list[i].increment_ages()
                print('No detections')

            # Stream results
            im0 = annotator.result()

            if opt.show_vid:
                cv2.imshow(str(p).encode("gbk").decode(errors="ignore"), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if opt.save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blazeface_model', type=str, default='weights/yolov5-blazeface.pt',
                        help='blazeface_model.pt path(s)')
    parser.add_argument('--max-stride', type=int, default=16, help='blazeface model max stride')
    parser.add_argument('--deep_sort_model', type=str, default='weights/osnet_x0_5_market1501.pth')
    parser.add_argument('--use_larger_box', action='store_true', default=False)
    parser.add_argument('--larger_box_coef', type=float, default=1.5)
    parser.add_argument('--face_crop_size', nargs='+', type=int, default=[72, 72],
                        help='face crop size h,w')  # height, width
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='face detect inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='face inference conf threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='face IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. cpu or cuda')
    parser.add_argument('--show-vid', action='store_true', help='display face track video results')
    parser.add_argument('--save-vid', action='store_true', help='save video face tracks results')
    parser.add_argument('--save-img', action='store_true', help='save raw and crop face tracks images')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", default=False, help="use FP16 half-precision inference")
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.face_crop_size *= 2 if len(opt.face_crop_size) == 1 else 1  # expand
    assert opt.face_crop_size[0] == opt.face_crop_size[1], 'face crop size width and height must be equal'

    detect(opt)
