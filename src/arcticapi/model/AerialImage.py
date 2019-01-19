import cv2
import numpy as np
import traceback
from PIL import Image as PILImage

from arcticapi.augmnetation import AugRgb, AugIR
from arcticapi.augmnetation.utils import get_image_size


class AerialImage():
    def __init__(self, path, type, camerapos):
        self.path = path
        self.type = type  # rgb, therm8, or therm16
        self.image = None  # not loaded
        self.camerapos = camerapos  # camera position
        self.hotspots = [] # hotspots in image
        self.w = None
        self.h = None
        self.file_exists = False

        try:
            self.w, self.h = get_image_size(self.path)
            if self.w is not None and self.h is not None:
                self.file_exists = True
        except Exception:
            traceback.print_exc()


    # Loads image to memory, returns true if success, false if not
    def load_image(self, colorJet=False):
        if self.image is not None:
            return True
        elif self.type == "rgb":
            self.image = cv2.imread(self.path)
        elif self.type == "thermal":
            self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        elif self.type == "ir":
            self.image = self.imreadIR(self.path, colorJet)
        ret = self.image is not None
        if not ret:
            print("Failed to load image " + self.path)
        self.h, self.w = self.image.shape[:2]
        return ret

    def free(self):
        del self.image
        self.image = None

    def tile(self):
        self.load_image()

    def imreadIR(self, fileIR):
        img = PILImage.open(fileIR)
        if img is None:
            return None
        img = np.array(img).astype(np.uint16)

        return img

    def generate_chips(self, cfg):
        """
        :type cfg: CropCfg
        """
        if cfg.imtype == "ir":
            AugIR.crop_ir_hotspot_8bit(cfg, self)
        elif cfg.imtype == "rgb":
            return AugRgb.prepare_chips(cfg, self)

    def getBboxes(self, cfg):
        iabboxs = []
        for hs in self.getHotSpots(cfg):
            iabboxs.append(hs.rgb_bb)
        return iabboxs

    def getHotSpots(self, cfg):
        hotspots = []
        for hs in self.hotspots:
            if hs.status == 'removed':
                continue
            # don't make crops or labels for bears
            if not cfg.make_bear and hs.classIndex == 3:
                continue
            # don't make crops or labels for anomalies
            if not cfg.make_anomaly and hs.classIndex == 4:
                continue
            if not hs.updated: # TODO: find a better spot for this and maybe save non-updated ones
                print("Hotspot " + hs.id + " not updated")
                continue
            hotspots.append(hs)
        return hotspots

