import os
import csv

from arcticapi import data_types, image_registration
from arcticapi.csv_parser import parse_hotspot
from arcticapi.crop import CropCfg

# This is the "model", it parses a NOAA seal formatted csv file and generates HotSpots - 1 per row.
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

    def register(self, id=None, showFigures=False, showImgs=False):
        if id is None:
            for hs in self.hsm.hotspots:
                image_registration.register_images(hs, showFigures, showImgs)
        else:
            hs = self.hsm.get_hs(id)
            if hs is not None:
                image_registration.register_images(hs, showFigures, showImgs)

    # Called with a CropCfg object it will crop and label all hotspots according to the given cfg
    def crop_label_all(self, cfg):
        """

        :type cfg: CropCfg
        """
        hs_ct = len(self.hsm.hotspots)
        print("processing " + str(hs_ct) + " hotspots")
        if not os.path.exists(cfg.out_dir):
            os.mkdir(cfg.out_dir)
        i = 0
        total_crops = 0
        classes = [0,0,0,0,0]
        for hs in self.hsm.hotspots:
            i += 1
            # don't make crops or labels for bears
            if not cfg.make_bear and hs.classIndex == 3:
                continue

            # don't make crops or labels for anomalies
            if not cfg.make_anomaly and hs.classIndex == 4:
                continue

            # combine all seals (Ringed, Bearded, UNK) into one class for training
            # this will be class 0
            if cfg.combine_seal:
                if hs.classIndex == 0 or hs.classIndex == 1 or hs.classIndex == 2:
                    hs.classIndex = 0
                if hs.classIndex == 3:
                    hs.classIndex = 1
                if hs.classIndex == 4:
                    hs.classIndex = 3

            if total_crops % 10 == 0:
                print("Cropping hotspot:" + str(hs.id) + " -" + str(
                    round((i + 0.0) / hs_ct, 2) * 100) + "% complete | " + str(total_crops) + "/" + str(hs_ct))

            total_crops += 1

            classes[hs.classIndex] += 1
            hs.genCropsAndLables(cfg)

        if cfg.combine_seal:
            print("Seals: " + str(classes[0]))
        else:
            print("Ringed Seals: " + str(classes[0]))
            print("Bearded Seals: " + str(classes[1]))
            print("NA Seals: " + str(classes[2]))
        print("Polar Bears: " + str(classes[3]))
        print("NA Animals: " + str(classes[4]))

