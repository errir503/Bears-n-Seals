import cv2
import numpy as np

from PIL import Image as PILImage
from arcticapi import crop
from arcticapi.crop import CropCfg

SpeciesList = ["Ringed Seal", "Bearded Seal", "UNK Seal", "Polar Bear", "NA"]


class HotSpot:
    def __init__(self, id, xpos, ypos, thumb_left, thumb_top, thumb_right, thumb_bottom, type, species_id, rgb,
                 thermal, ir, timestamp, project_name, aircraft):
        self.id = id  # id_hotspot
        self.thermal_loc = (xpos, ypos)  # location in thermal image
        # Bounding box
        self.rgb_bb_l = thumb_left
        self.rgb_bb_r = thumb_right
        self.rgb_bb_t = thumb_top
        self.rgb_bb_b = thumb_bottom
        self.type = type
        self.species = species_id
        self.classIndex = SpeciesList.index(species_id)
        self.rgb = rgb
        self.thermal = thermal
        self.ir = ir
        self.timestamp = timestamp
        self.project_name = project_name
        self.aircraft = aircraft

    def load_all(self):
        if self.thermal.load_image() and self.rgb.load_image() and self.ir.load_image():
            return True
        else:
            print("Skipped " + self.id)
            return False

    def free_all(self):
        self.rgb.free()
        self.ir.free()
        self.thermal.free()

    def getRGBCenterPt(self):
        x = self.rgb_bb_l + ((self.rgb_bb_r - self.rgb_bb_l) / 2)
        y = self.rgb_bb_t + ((self.rgb_bb_b - self.rgb_bb_t) / 2)
        return (x, y)

    def genCropsAndLables(self, cfg):
        """
        :type cfg: CropCfg
        """
        if cfg.imtype == "ir":
            crop.crop_ir_hotspot_8bit(cfg, self)
        elif cfg.imtype == "rgb":
            crop.crop_rgb_hotspot(cfg, self)

class Image():
    def __init__(self, path, type, camerapos):
        self.path = path
        self.type = type  # rgb, therm8, or therm16
        self.image = None  # not loaded
        self.camerapos = camerapos  # camera position

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
        return ret

    def free(self):
        del self.image
        self.image = None

    def tile(self):
        self.load_image()

    def imreadIR(self, fileIR, colorJet=False):
        # return norm.raw16bit(fileIR)
        # img = norm.normalize_percentile2(fileIR, False)
        # img = norm.normalize_ir_global(self.camerapos, fileIR, False).astype(np.uint8)
        # img = norm.norm(fileIR, False).astype(np.uint8)
        # return imgNorm.astype(np.uint8), imgGlobalNorm.astype(np.uint8), imgLocalNorm.astype(np.uint8), anyDepth
        img = PILImage.open(fileIR)
        if img is None:
            return None
        img = np.array(img).astype(np.uint16)

        return img


#
class HotSpotMap:
    def __init__(self):
        self.images = {}
        self.hs_id_to_idx = {}
        self.hotspots = []
        return

    def add(self, hotspot):
        rgb = hotspot.rgb
        if rgb.path not in self.images:
            self.images[rgb.path] = []

        thermal = hotspot.thermal
        if thermal.path not in self.images:
            self.images[thermal.path] = []

        ir = hotspot.ir
        if ir.path not in self.images:
            self.images[ir.path] = []

        self.images[rgb.path].append(len(self.hotspots))
        self.images[thermal.path].append(len(self.hotspots))
        self.images[ir.path].append(len(self.hotspots))

        self.hs_id_to_idx[hotspot.id] = len(self.hotspots)
        self.hotspots.append(hotspot)
        return

    def get_hs(self, id):
        if str(id) in self.hs_id_to_idx:
            return self.hotspots[self.hs_id_to_idx[str(id)]]
        print("No HotSpot with id: " + str(id))
        return None
