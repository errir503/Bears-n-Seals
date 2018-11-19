import os
import csv

from arcticapi import data_types, image_registration
from arcticapi.label_parser import parse_hotspot
from arcticapi.crop import CropCfg


class ArcticApi:
    def __init__(self, csv_path, im_path):
        rows = list()

        f = open(csv_path, 'r')
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
        f.close()
        del rows[0]  # remove col headers


        hsm = data_types.HotSpotMap()

        for row in rows:
            hotspot = parse_hotspot(row, im_path)
            hsm.add(hotspot)

        self.hsm = hsm
        del rows

    def get_hotspots(self):
        return self.hsm

    def register(self, id=None, showFigures=False, showImgs=False):
        if id is None:
            for hs in self.hsm.hotspots:
                image_registration.register_images(hs, showFigures, showImgs)
        else:
            hs = self.hsm.get_hs(id)
            if hs is not None:
                image_registration.register_images(hs, showFigures, showImgs)

    def crop_label_all(self, cfg):
        """

        :type cfg: CropCfg
        """
        hs_ct = len(self.hsm.hotspots)
        print("processing " + str(hs_ct) + " hotspots")
        print(cfg.tostr())

        if not os.path.exists(cfg.out_dir):
            os.mkdir(cfg.out_dir)
        i = 0
        total_crops = 0
        classes = [0,0,0,0,0]
        for hs in self.hsm.hotspots:
            i += 1
            if not cfg.make_bear and hs.classIndex == 3:
                continue

            if not cfg.make_anomaly and hs.classIndex == 4:
                continue



            if cfg.combine_seal:
                if hs.classIndex == 0 or hs.classIndex == 1 or hs.classIndex == 2:
                    hs.classIndex = 0
            if total_crops % 10 == 0:
                print("Cropping hotspot:" + str(hs.id) + " -" + str(
                    round((i + 0.0) / hs_ct, 2) * 100) + "% complete | " + str(total_crops) + "/" + str(hs_ct))

            total_crops += 1

            classes[hs.classIndex] += 1
            hs.genCropsAndLables(cfg)

        if cfg.combine_seals:
            print("Se.ls: " + str(classes[0]))
        else:
            print("Ringed Seals: " + str(classes[0]))
            print("Bearded Seals: " + str(classes[1]))
            print("NA Seals: " + str(classes[2]))
        print("Polar Bears: " + str(classes[3]))
        print("NA Animals: " + str(classes[4]))
