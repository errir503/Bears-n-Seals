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
        self.center_x = thumb_left + ((thumb_right - thumb_left) / 2)
        self.center_y = thumb_bottom + ((thumb_top - thumb_bottom) / 2)
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

    def getIRCenterPt(self):
        return (self.center_x, self.center_y)
