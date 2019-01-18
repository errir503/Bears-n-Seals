import random
import copy
import cv2
import numpy as np
import imgaug as ia

from imgaug import augmenters as iaa
from arcticapi.augmnetation.utils import getYoloFromRect
from arcticapi.visuals import drawBBoxYolo


class TrainingChip():
    # constructs a training chip with imgaug bounding boxes and saves the image to filename so that it is not
    # stored in memory.  later on can use load, augmentations, and save for the data augmentation step
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
        self.save_image() # save the image

    # loads the chip
    def load(self):
        if self.image is None:
            self.image = cv2.imread(self.filename + ".jpg")

    # free the chip from memory
    def free(self):
        del self.image
        self.image = None

    # save the
    def save_image(self):
        cv2.imwrite(self.filename + ".jpg", self.image)

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

                if self.cfg.debug:  # draws same as yolo so guaranteed to show if labels are correct
                    drawBBoxYolo(self.image, x, y, w, h)

        self.save_image()

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

    # adds random values between min and max to the hue and saturation of the image
    # if per_channel is true then adds independently per channel and the same value for all pixels within that channel
    def color_change(self, min, max, per_channel = False):
        img = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        self.image = iaa.AddToHueAndSaturation((min, max), per_channel=per_channel).augment_image(img)


    # rotate image either 0, 90, 180, or 290 degrees
    def rotate(self):
        rotations = [0, 90, 180, 270]
        seq = iaa.Sequential([
            iaa.Affine(rotate=rotations),
        ])
        seq_det = seq.to_deterministic()
        im = seq_det.augment_images([self.image])[0]
        self.image = np.ascontiguousarray(im, dtype=np.uint8)
        new = seq_det.augment_bounding_boxes([self.bboxes])[0]
        for i in range(len(self.bboxes.bounding_boxes)):
            new.bounding_boxes[i].hsId = self.bboxes.bounding_boxes[i].hsId

        self.bboxes = new

    # 50% chance to flip horizontally and 50% chance to flip vertically
    def flip(self):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
            ])

        seq_det = seq.to_deterministic()
        im = seq_det.augment_images([self.image])[0]
        self.image = np.ascontiguousarray(im, dtype=np.uint8)
        new = seq_det.augment_bounding_boxes([self.bboxes])[0]
        for i in range(len(self.bboxes.bounding_boxes)):
            new.bounding_boxes[i].hsId = self.bboxes.bounding_boxes[i].hsId

        self.bboxes = new

    def copy(self):
        return copy.deepcopy(self)