import cv2
import numpy as np

import normalizer as norm

SpeciesList = ["Ringed Seal", "Bearded Seal", "Polar Bear", "UNK Seal", "NA"]


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

class Image():
    def __init__(self, path, type, camerapos):
        self.path = path
        self.type = type  # rgb, therm8, or therm16
        self.image = None  # not loaded
        self.camerapos = camerapos  # camera position

    # Loads image to memory, returns true if success, false if not
    def load_image(self, colorJet = False):
        if self.image is not None:
            return True
        elif self.type == "rgb":
            self.image = cv2.imread(self.path)
        elif self.type == "thermal":
            self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        elif self.type == "ir":
            self.image = self.imreadIR(self.path, colorJet)
        return self.image is not None

    def free(self):
        del self.image
        self.image = None

    def tile(self):
        self.load_image()

    def imreadIR(self, fileIR, colorJet, percent=0.005):
        anyDepth = cv2.imread(fileIR, cv2.IMREAD_ANYDEPTH)
        if (not anyDepth is None):
            imgGlobalNorm = norm.normalize_ir_global(self.camerapos, fileIR)
            imgLocalNorm = norm.normalize_ir_local(self.camerapos, fileIR)
            imgNorm = np.floor((anyDepth - np.percentile(anyDepth, percent)) / (
                        np.percentile(anyDepth, 100 - percent) - np.percentile(anyDepth, percent)) * 256)
            if colorJet:
                imgNorm = cv2.applyColorMap(imgNorm.astype(np.uint8), cv2.COLORMAP_HSV)
                anyDepth = cv2.applyColorMap(anyDepth.astype(np.uint8), cv2.COLORMAP_HSV)
                imgGlobalNorm = cv2.applyColorMap(imgGlobalNorm.astype(np.uint8), cv2.COLORMAP_HSV)
                imgLocalNorm = cv2.applyColorMap(imgLocalNorm.astype(np.uint8), cv2.COLORMAP_HSV)
            return imgNorm.astype(np.uint8), anyDepth, imgGlobalNorm.astype(np.uint8), imgLocalNorm.astype(np.uint8)
        return None


#
class HotSpotMap:
    def __init__(self):
        self.images = {}
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

        self.hotspots.append(hotspot)
        return


