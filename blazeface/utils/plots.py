"""
Plotting utils
"""

import cv2
import numpy as np


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class Annotator:

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        self.tf = max(self.lw - 1, 1)  # font thickness

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=self.tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                        thickness=self.tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def ppg_label(self, ppgs, location='left', radio=0.7, box_width_radio=0.05, box_height_radio=0.2):
        if location == 'left' or location == 'right':
            displayWidth, displayHeight = int(self.im.shape[1] * radio), self.im.shape[0]
            display_box_weight, display_box_height = displayWidth, (displayHeight // len(ppgs) if len(ppgs) else None)
        else:
            displayWidth, displayHeight = self.im.shape[1], int(self.im.shape[0] * radio)
            display_box_weight, display_box_height = (displayWidth // len(ppgs) if len(ppgs) else None), displayHeight
        drawArea = np.full((displayHeight, displayWidth, 3), 255, dtype=np.uint8)  # white background

        for i, (id, ppg) in enumerate(ppgs.items()):
            vmin, vmax = ppg.min(), ppg.max()
            heightMult, widthMult = display_box_height * (1 - 2 * box_height_radio) / (
                    vmax - vmin), display_box_weight * (
                                            1 - 2 * box_width_radio) / len(ppg)
            w, h = cv2.getTextSize(id, 0, fontScale=self.lw / 3, thickness=self.tf)[0]
            if location == 'left' or location == 'right':
                cv2.putText(drawArea, id, (
                    int((drawArea.shape[1] - w) / 2),
                    int(display_box_height * i + (box_height_radio * display_box_height + h) / 2)),
                            0,
                            self.lw / 3, (192, 192, 192),
                            thickness=self.tf, lineType=cv2.LINE_AA)

                display_x, display_y = box_width_radio * display_box_weight, (
                        i + 1 - box_height_radio) * display_box_height
                if i != (len(ppgs) - 1):
                    cv2.line(drawArea, (0, (i + 1) * display_box_height),
                             (display_box_weight, (i + 1) * display_box_height),
                             (255, 0, 0), 1)
            else:
                cv2.putText(drawArea, id, (
                    int(i * display_box_weight + (display_box_weight - w) / 2),
                    int((box_height_radio * display_box_height + h) / 2)),
                            0,
                            self.lw / 3, (192, 192, 192),
                            thickness=self.tf, lineType=cv2.LINE_AA)

                display_x, display_y = (i + box_width_radio) * display_box_weight, (
                        1 - box_height_radio) * display_box_height
                if i != (len(ppgs) - 1):
                    cv2.line(drawArea, ((i + 1) * display_box_weight, 0),
                             ((i + 1) * display_box_weight, display_box_height),
                             (255, 0, 0), 1)

            p1 = None
            for index, point in enumerate(ppg):
                p2 = (int(display_x + index * widthMult), int(display_y - (point - vmin) * heightMult))
                if p1 is None:
                    p1 = p2
                    continue
                cv2.line(drawArea, p1, p2, (0, 0, 255), 1)
                p1 = p2

        self.im = np.concatenate(
            ([self.im, drawArea] if location == 'right' or location == 'bottom' else [drawArea, self.im]),
            axis=(1 if location == 'left' or location == 'right' else 0))

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)
