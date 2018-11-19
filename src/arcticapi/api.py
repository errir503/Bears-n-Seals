import os
import csv

from arcticapi import data_types, image_registration
from arcticapi.label_parser import parse_hotspot


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
        ringed_seal_ct = 0
        bearded_seal_ct = 0
        polar_bear_ct = 0
        na_seal_ct = 0
        na_animal_ct = 0
        for row in rows:
            hotspot = parse_hotspot(row, im_path)
            if hotspot.classIndex == 0:
                ringed_seal_ct += 1
            elif hotspot.classIndex == 1:
                bearded_seal_ct += 1
            elif hotspot.classIndex == 2:
                na_seal_ct += 1
            elif hotspot.classIndex == 3:
                polar_bear_ct += 1
            elif hotspot.classIndex == 4:
                na_animal_ct += 1

            hsm.add(hotspot)

        self.hsm = hsm
        print("Ringed Seals: " + str(ringed_seal_ct))
        print("Bearded Seals: " + str(bearded_seal_ct))
        print("Polar Bears: " + str(polar_bear_ct))
        print("NA Seals: " + str(na_seal_ct))
        print("NA Animals: " + str(na_animal_ct))
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

    def crop_label_all(self, out_dir, width_bb, minShift, maxShift, crop_size, label, combine_seals, train_bear, train_anomaly):
        hs_ct = len(self.hsm.hotspots)
        print("Processing " + str(hs_ct) + " hotspots")
        print("Combining all seals into 1 seal class = " + str(combine_seals))
        print("Generating polar bear crops = " + str(train_bear))
        print("Generating anomaly crops crops = " + str(train_anomaly))
        print("")
        print("Bounding box w/h: " + str(width_bb))
        print("Crop size: " + str(crop_size))
        print("")
        print("Output to crops and labels saving to: " + out_dir)
        print("Training label list saving to: " + label)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        i = 0
        total_crops = 0
        for hs in self.hsm.hotspots:
            i += 1
            if not train_bear and hs.classIndex == 3:
                continue

            if not train_anomaly and hs.classIndex == 4:
                continue



            if combine_seals:
                if hs.classIndex == 0 or hs.classIndex == 1 or hs.classIndex == 2:
                    hs.classIndex = 0
            if total_crops % 10 == 0:
                print("Cropping hotspot:" + str(hs.id) + " -" + str(
                    round((i + 0.0) / hs_ct, 2) * 100) + "% complete, total " + str(total_crops))

            total_crops += 1


            hs.genCropsAndLables(out_dir, width_bb, minShift, maxShift, crop_size, label)
