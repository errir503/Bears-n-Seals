import os

import cv2
import numpy as np
import traceback
from PIL import Image as PILImage

from src.arcticapi.augmnetation import AugRgb, AugIR
from src.arcticapi.augmnetation.utils import get_image_size
from src.arcticapi.visuals import *


class AerialImage():
    def __init__(self, path, type, camerapos, fog = "NA"):
        self.path = path
        self.type = type  # rgb, therm8, or therm16
        self.image = None  # not loaded
        self.camerapos = camerapos  # camera position
        self.hotspots = [] # hotspots in image
        self.w = None
        self.h = None
        self.file_exists = False
        self.fog = fog
        self.chips = None

        try:
            self.w, self.h = get_image_size(self.path)
            if self.w is not None and self.h is not None:
                self.file_exists = True
        except Exception:
            traceback.print_exc()


    # Loads image to memory, returns true if success, false if not
    def load_image(self):
        if self.image is not None:
            return True
        elif self.type == "rgb":
            self.image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        elif self.type == "thermal":
            self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        elif self.type == "ir":
            self.image = cv2.imread(self.path)
            # self.image = self.imreadIR(self.path)
        ret = self.image is not None
        if not ret:
            print("Failed to load image " + self.path)
            return ret
        self.h, self.w = self.image.shape[:2]
        return ret

    def free(self):
        del self.image
        self.image = None

    def tile(self):
        self.load_image()

    def imreadIR(self, fileIR):
        try:
            img = PILImage.open(fileIR)
        except:
            return None
        if img is None:
            return None
        img = np.array(img).astype(np.uint16)

        return img

    def generate_chips(self, cfg):
        """
        :type cfg: CropCfg
        """
        if cfg.imtype == "ir":
            return AugIR.crop_ir_hotspot_8bit(cfg, self)
        elif cfg.imtype == "rgb":
            if not self.file_exists:
                return []
            bounding_boxes = self.getBboxesForTraining(cfg)
            return AugRgb.prepare_chips(cfg, self, bounding_boxes)

    def generate_all_chips(self, cfg):
        """
        :type cfg: CropCfg
        """
        if cfg.imtype == "ir":
            AugIR.crop_ir_hotspot_8bit(cfg, self)
        elif cfg.imtype == "rgb":
            if not self.file_exists:
                return []
            bounding_boxes = self.getBboxesForReLabeling()
            return AugRgb.prepare_chips(cfg, self, bounding_boxes)

    def getBboxesForTraining(self, cfg):
        iabboxs = []
        # for hs in self.getHotspotsForTraining(cfg):
        #     iabboxs.append(hs.rgb_bb)
        for hs in self.getHotspotsForTraining(cfg):
            iabboxs.append(hs.rgb_bb)
        return iabboxs

    def getBboxesForReLabeling(self):
        iabboxs = []
        for hs in self.hotspots:
            if hs.updated_left != -1:
                iabboxs.append(hs.rgb_bb)
                continue
            new_box = hs.rgb_bb.extend(all_sides=-200)
            new_box.hsId = hs.rgb_bb.hsId
            new_box.label = hs.rgb_bb.label
            iabboxs.append(new_box)
        return iabboxs

    def getHotspotsForReLabeling(self, cfg):
        hotspots = []
        for hs in self.hotspots:
            if hs.filterClass(cfg):
                continue
            if hs.isStatusRemoved() or not hs.updated:
                hotspots.append(hs)
        return hotspots

    def getHotspotsForTraining(self, cfg):
        hotspots = []
        for hs in self.hotspots:
            if hs.isStatusRemoved():
                continue
            if hs.filterClass(cfg):
                continue
            if not hs.updated:
                if cfg.debug:
                    print("Hotspot " + hs.id + " not updated")
                continue
            hotspots.append(hs)
        return hotspots

    def saveLabels(self, cfg):
        if cfg.debug:
            if not self.load_image():
                print("Failed to load")
                return

        if self.w is None or self.h is None:
            if not self.load_image():
                print("Failed to load")
                return
            else:
                self.h = self.image.shape[0]
                self.w = self.image.shape[1]


        file_name = cfg.im_dir + os.path.splitext(os.path.basename(self.path))[0]
        added = 0
        for hs in self.hotspots:
            if hs.isStatusRemoved():
                continue
            if hs.filterClass(cfg):
                continue
            if not hs.updated:
                continue
            with open(file_name + ".txt", 'a') as file:
                x, y, w, h = hs.getYoloBBox(np.zeros([self.h,self.w,3],dtype=np.uint8))
                classIndex = hs.classIndex

                if cfg.combine_seal and (classIndex == 0 or classIndex == 1 or classIndex == 2):
                    classIndex = 0

                yoloLabel = (classIndex, x, y, w, h)
                file.write(" ".join([str(i) for i in yoloLabel]) + "\n")

            if cfg.debug:  # draws same as yolo so guaranteed to show if labels are correct
                drawBBoxYolo(self.image, x, y, w, h, classIndex)
            added += 1
        if added > 0:
            if cfg.debug:
                cv2.imwrite(file_name + "-debug.jpg", self.image)
            with open(cfg.label, 'a') as file:
                file.write(cfg.im_dir + os.path.basename(self.path) + "\n")



