import os
import csv
import random

from arcticapi.augmnetation.utils import write_label
from arcticapi.csv_parser import parse_hotspot
from arcticapi.registration import image_registration

from arcticapi.model.HotSpotMap import HotSpotMap

# This is the controller of the api which is created using the path of the full resolution image directory
# and the path to the NOAA hotspot csv file.
# Once created contains a list of hotspots and a list of AerialImages which contain all information about the image,
# and hotspots within that image
class ArcticApi:
    def __init__(self, csv_path, im_path):
        rows = list()

        f = open(csv_path, 'r')
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
        f.close()
        header = rows[0]
        del rows[0]  # remove col headers

        hsm = HotSpotMap()
        images = {}
        for row in rows:
            hotspot = parse_hotspot(row, im_path)
            hsm.add(hotspot)
            if not hotspot.rgb.path in images:
                images[hotspot.rgb.path] = hotspot.rgb
            images[hotspot.rgb.path].hotspots.append(hotspot)

        self.csvheader = header
        self.images = images
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

    def crop_label_images(self, cfg):
        img_ct = len(self.images)
        print("processing " + str(img_ct) + " images")
        if not os.path.exists(cfg.out_dir):
            os.mkdir(cfg.out_dir)

        label_base = cfg.label.split(".")[0]
        chips = []
        for image_path in self.images:
            if cfg.debug and len(chips) > 100:
                break
            chips = chips + self.images[image_path].generate_chips(cfg)

        random.shuffle(chips)
        x = int(len(chips)/4) * 3  # 3/4 train 1/4 test
        train = chips[:x]
        test = chips[x:]

        for chip in train:
            chip.load() # load image
            copy = chip.copy()
            copy.load()
            copy.filename = copy.filename + "_b"
            chips = [chip, copy]
            for c in chips:
                # augmentations
                c.color_change(-10, 10, False)
                c.extend(5)
                c.flip()
                c.rotate()
                c.save() # save image
                write_label(c.filename + ".jpg", label_base + "_train.txt")
            chip.free()  # free image
            copy.free()  # free image

        for chip in test:
            chip.load() # load image
            chip.extend(5)
            chip.save() # save image and labels
            chip.free() # free image
            write_label(chip.filename + ".jpg", label_base + "_test.txt")



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

            if cfg.combine_all:
                hs.classIndex = 0
            else:
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
            hs.generate_chips(cfg)

        if cfg.combine_all:
            print("Hotspots: " + str(classes[0]))
        elif cfg.combine_seal:
            print("Seals: " + str(classes[0]))
            print("Polar Bears: " + str(classes[3]))
            print("NA Animals: " + str(classes[4]))
        else:
            print("Ringed Seals: " + str(classes[0]))
            print("Bearded Seals: " + str(classes[1]))
            print("NA Seals: " + str(classes[2]))
            print("Polar Bears: " + str(classes[3]))
            print("NA Animals: " + str(classes[4]))

    # Save all hotspots unfiltered in the standard seal csv format
    def saveHotspotsToCSV(self, out_file):
        with open(out_file, 'w') as temp_file:
            temp_file.write(",".join(self.csvheader) + "\n")
            for hs in self.hsm.hotspots:
                newrowtxt = hs.toCSVRow()
                temp_file.write(newrowtxt)


