import cv2

SpeciesList = ["Ringed Seal", "Bearded Seal", "Polar Bear", "UNK Seal", "NA"]


class HotSpot:
    def __init__(self, id, xpos, ypos, thumb_left, thumb_top, thumb_right, thumb_bottom, type, species_id, rgb,
                 thermal8, thermal16, timestamp):
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
        self.therm8 = thermal8
        self.therm16 = thermal16
        self.timestamp = timestamp

class Image():
    def __init__(self, path, type):
        self.path = path
        self.type = type  # rgb, therm8, or therm16
        self.image = None  # not loaded

    # Loads image to memory, returns true if success, false if not
    def load_image(self):
        if type == "rgb":
            self.image = cv2.imread(self.path)
        else:
            self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        return self.image is not None

    def free(self):
        del self.image
        self.image = None

    def tile(self):
        self.load_image()

# Not totally decided how this class will function but essentially maps hotspots to their
# respective images and visa-versa
class HotSpotMap:
    def __init__(self):
        self.images = {}
        self.hotspots = {}
        return

    def add(self, images, hotspot):
        for img in images:
            if (self.images.get(img.path) == None):
                self.images[img.path] = list()
            self.hotspots[hotspot.id] = hotspot
            self.images[img.path].append(hotspot.id)
        return


