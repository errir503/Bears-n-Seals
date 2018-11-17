import os
import sys
import csv

import cv2

from arcticapi import data_types, image_registration
from arcticapi.label_parser import parse_hotspot

class ArcticApi:
    def __init__(self, csv_path):
        rows = list()

        f = open(sys.argv[1], 'r')
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
        f.close()
        del rows[0]  # remove col headers

        global res_path
        res_path = sys.argv[2]
        hsm = data_types.HotSpotMap()

        for row in rows:
            hotspot = parse_hotspot(row, res_path)
            hsm.add(hotspot)

        self.hsm = hsm
        del rows

    def get_hotspots(self):
        return self.hsm


    def register(self, id = None, showFigures=False, showImgs=False):
        if id is None:
            for hs in self.hsm.hotspots:
                image_registration.register_images(hs, showFigures, showImgs)
        else:
            hs = self.hsm.get_hs(id)
            if hs is not None:
                image_registration.register_images(hs, showFigures, showImgs)




    def prep_labels(self, crop_offset, out_dir):
        # Check if output directory exists, if not create
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        i = 0
        for hs in self.hsm.hotspots:
            print("Cropping hotspot:" + str(hs.id) + " -" + str(round((i + 0.0) / len(self.hsm.hotspots), 2))) + "% complete"
            i += 1
            self.prep_label(out_dir, crop_offset, hs.id, True)

