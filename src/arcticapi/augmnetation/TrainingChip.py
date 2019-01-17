import random

import cv2
import numpy as np
import imgaug as ia

from arcticapi.augmnetation.utils import write_label, getYoloFromRect
from arcticapi.visuals import drawBBoxYolo


class TrainingChip():
    def __init__(self, image, cfg, imgpath, bboxes, crops):
        self.image = image
        # format [(classname, x, y, w, h, hotspotId),...] in yolo with additional hotspot id
        self.cfg = cfg
        self.crops = crops # (topcrop, bottomcrop, leftcrop, rightcrop)
        self.imgpath = imgpath
        ids = [x.hsId for x in bboxes]
        self.filename = cfg.out_dir + "crop_" + "_".join(ids)
        boxes = []
        for bbox in bboxes:
            new = bbox.cut_out_of_image(image)
            new.hsId = bbox.hsId
            boxes.append(new)

        self.bboxes = ia.BoundingBoxesOnImage(boxes, shape=image.shape)

    def save(self):
        # if no labels, still a training image save with empty label file for darknet
        if len(self.bboxes.bounding_boxes) == 0:
            # write_label(self.filename + ".jpg", self.cfg.label)
            # open(self.filename + ".txt", 'a').close()
            # cv2.imwrite(self.filename + ".jpg", self.image)
            return

        # Generate trainin label
        for bbs in self.bboxes.bounding_boxes:
            with open(self.filename + ".txt", 'a') as file:
                classIndex = bbs.label

                if self.cfg.combine_seal and (classIndex == 0 or classIndex == 1 or classIndex == 2):
                    bbs.label = 0

                x,y,w,h = getYoloFromRect(self.bboxes.height, self.bboxes.width, bbs.x1, bbs.y1, bbs.x2, bbs.y2)
                yoloLabel = (bbs.hsId, bbs.label, x, y, w, h)
                file.write(" ".join([str(i) for i in yoloLabel[1:]]) + "\n")
                # create 2label file which allows to use the bounding box labeler tool to
                # go through crops and to re-label.  .2label file formatted as
                # hsid classid x y w h topcrop bottomcrop leftcrop rightcrop
                # (last 4 are tile's location in original image)
                with open(self.filename + ".2label", 'a') as file:
                    file.write(" ".join([str(i) for i in yoloLabel]) + " " +
                           " ".join([str(i) for i in self.crops]) + "\n")

                if self.cfg.debug:  # draws same as yolo so will prove labels are correct
                    drawBBoxYolo(self.image, x, y, w, h)

        cv2.imwrite(self.filename + ".jpg", self.image)
        write_label(self.filename + ".jpg", self.cfg.label)

    def random_hue_adjustment(self, ratio):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        ratio = random.uniform(1-ratio, 1 + ratio)
        hsv[:,:,2] =  np.clip(hsv[:,:,2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



    # extend the size of all bbox sides by px
    def extend(self, px):
        new = []
        for bbox in self.bboxes.bounding_boxes:
            new_box = bbox.extend(all_sides=px)
            new_box.hsId = bbox.hsId
            new_box.label = bbox.label
            new.append(new_box)
        self.bboxes = ia.BoundingBoxesOnImage(new, shape=self.image.shape)

    def augment(self):
        self.image = ia.imresize_single_image(self.image, (320, 320))
        self.bbs = self.bbs.on(self.image)
        if self.cfg.debug:
            image_bbs = self.bbs.draw_on_image(self.image, thickness=2)
            bbs_rescaled = self.bbs.draw_on_image(self.image, thickness=2)
        return bbs_rescaled