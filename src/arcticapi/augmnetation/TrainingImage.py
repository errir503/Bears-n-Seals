import random

import cv2
import numpy as np


from arcticapi.augmnetation.utils import write_label


class TrainingImage():
    def __init__(self, image, cfg, filename):
        self.image = image
        # format [(classname, x, y, w, h),...] in yolo format
        self.bboxes = []
        self.cfg = cfg
        self.filename = filename


    def save(self):
        # if no labels, still a training image save with empty label file for darknet
        if len(self.bboxes) == 0:
            write_label(self.filename + ".jpg", self.cfg.label)
            open(self.filename + ".txt", 'a').close()
            cv2.imwrite(self.filename + ".jpg", self.image)
            return

        # Generate trainin label
        for box in self.bboxes:
            with open(self.filename + ".txt", 'a') as file:
                classIndex = box[0]

                if self.cfg.combine_seal:
                    if classIndex == 0 or classIndex == 1 or classIndex == 2:
                        box[0] = 0

                file.write(" ".join([str(i) for i in box]) + "\n")
                if self.cfg.debug:  # draws same as yolo so will prove labels are correct
                    (imw,imh, imc) = self.image.shape
                    (classId, x, y, w, h) = box

                    x = int(x * imw)
                    y = int(y * imh)
                    w = int(w * imw)
                    h = int(h * imh)
                    cv2.circle(self.image, (x, y), 5, (0, 255, 0), 2)
                    cv2.rectangle(self.image, (x - w / 2, y - h / 2),
                                  (x + w / 2, y + h / 2),
                                  (0, 255, 0), 2)  # draw rect

        cv2.imwrite(self.filename + ".jpg", self.image)
        write_label(self.filename + ".jpg", self.cfg.label)

    def random_hue_adjustment(self, ratio):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        ratio = random.uniform(1-ratio, 1 + ratio)
        hsv[:,:,2] =  np.clip(hsv[:,:,2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)