import random

import cv2
import numpy as np


from arcticapi.augmnetation.utils import write_label
from arcticapi.visuals import drawBBoxYolo


class TrainingImage():
    def __init__(self, image, cfg, imgpath, bboxes, crops):
        self.image = image
        # format [(classname, x, y, w, h, hotspotId),...] in yolo with additional hotspot id
        self.bboxes = bboxes
        self.cfg = cfg
        self.crops = crops # (topcrop, bottomcrop, leftcrop, rightcrop)
        self.imgpath = imgpath
        ids = [x[0] for x in self.bboxes]
        self.filename = cfg.out_dir + "crop_" + "_".join(ids)


    def save(self):
        # if no labels, still a training image save with empty label file for darknet
        if len(self.bboxes) == 0:
            write_label(self.filename + ".jpg", self.cfg.label)
            write_label(" ".join([str(i) for i in self.crops]), self.cfg.label + "_orig")
            open(self.filename + ".txt", 'a').close()
            cv2.imwrite(self.filename + ".jpg", self.image)
            return

        # Generate trainin label
        for box in self.bboxes:
            with open(self.filename + ".txt", 'a') as file:
                classIndex = box[0]

                if self.cfg.combine_seal:
                    if classIndex == 0 or classIndex == 1 or classIndex == 2:
                        x,y,w,h =box[2:]
                        box = (box[0], 0, x,y,w,h )

                file.write(" ".join([str(i) for i in box[1:]]) + "\n")
                # create 2label file which allows to use the bounding box labeler tool to
                # go through crops and to re-label.  .2label file formatted as
                # hsid classid x y w h topcrop bottomcrop leftcrop rightcrop
                # (last 4 are tile's location in original image)
                with open(self.filename + ".2label", 'a') as file:
                    file.write(" ".join([str(i) for i in box]) + " " +
                           " ".join([str(i) for i in self.crops]) + "\n")

                if self.cfg.debug:  # draws same as yolo so will prove labels are correct
                    (hsId, classId, x, y, w, h) = box
                    drawBBoxYolo(self.image, x, y, w, h)

        cv2.imwrite(self.filename + ".jpg", self.image)
        write_label(self.filename + ".jpg", self.cfg.label)

    def random_hue_adjustment(self, ratio):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        ratio = random.uniform(1-ratio, 1 + ratio)
        hsv[:,:,2] =  np.clip(hsv[:,:,2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)