import os
import csv
import random
import sys

from arcticapi.augmnetation import AugRgb
from arcticapi.augmnetation.utils import write_label
from arcticapi.csv_parser import parse_hotspot
from arcticapi.registration import image_registration
from arcticapi.model.HotSpotMap import HotSpotMap
from arcticapi.visuals import print_loading_bar

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
        del rows[0]  # remove col headers

        hsm = HotSpotMap()
        images = {}
        for row in rows:
            hotspot = parse_hotspot(row, im_path)
            hsm.add(hotspot)
            if not hotspot.rgb.path in images:
                images[hotspot.rgb.path] = hotspot.rgb
            images[hotspot.rgb.path].hotspots.append(hotspot)

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

    def crop_for_labeling(self,cfg):
        img_ct = len(self.images)
        print("processing " + str(img_ct) + " images")
        if not os.path.exists(cfg.out_dir):
            os.mkdir(cfg.out_dir)

        label_base = cfg.label.split(".")[0]
        chips = []
        for image_path in self.images:
            chips = chips + self.images[image_path].generate_chips(cfg, False)
        print("\nOriginal Stats:")
        AugRgb.print_bbox_stats(chips)

        for idx, chip in enumerate(chips):
            print_loading_bar(((idx + 0.0) / len(chips)) * 100.0)
            if not chip.load():
                print("Chip not loaded in api.py :(")
                continue
            chip.load()  # load image
            chip.save()  # save image and labels
            chip.free()  # free image
            write_label(chip.filename + ".jpg", label_base + "_test.txt")

        print("COMPLETE")
        return


    def generate_training_set(self, cfg):
        img_ct = len(self.images)
        print("processing " + str(img_ct) + " images")
        if not os.path.exists(cfg.out_dir):
            os.mkdir(cfg.out_dir)

        label_base = cfg.label.split(".")[0]
        chips = []
        for image_path in self.images:
            chips = chips + self.images[image_path].generate_chips(cfg, True)

        print("\nOriginal Stats:")
        AugRgb.print_bbox_stats(chips)
        print

        # Test/Train split
        train, test = AugRgb.test_train_split(chips)
        print("Attempting to make classes of similar size")
        train = AugRgb.equalize_classes(train)
        print

        print("Training set stats:")
        AugRgb.print_bbox_stats(train)
        print("Generating training set...")
        for idx, c in enumerate(train):
            pct = ((idx + 0.0) / len(train)) * 100.0
            sys.stdout.write("\r|%-73s| %3d%%" % ('#' * int(pct * .73), pct))
            # models tend to struggle with larger seals so allow more zoom in than out
            if not c.load(random.uniform(-.05, 0.2)):
                print("Chip not loaded in api.py :( %s" % c.filename)
                continue
            # augmentations
            c.color_change(-10, 10, False)
            c.extend(5)
            c.flip()
            c.rotate()
            c.save()  # save image
            write_label(c.filename + ".jpg", label_base + "_train.txt")
            c.free()

        print("Testing set stats:")
        AugRgb.print_bbox_stats(train)
        print("Generating test set...")
        for idx, chip in enumerate(test):
            print_loading_bar(((idx + 0.0) / len(test)) * 100.0)
            if not chip.load():
                print("Chip not loaded in api.py :(")
                continue
            chip.load()  # load image
            chip.extend(5)
            chip.save()  # save image and labels
            chip.free()  # free image
            write_label(chip.filename + ".jpg", label_base + "_test.txt")

        print("COMPLETE")
        return

    # Save all hotspots unfiltered in the standard seal csv format
    def saveHotspotsToCSV(self, out_file, header):
        with open(out_file, 'w') as temp_file:
            temp_file.write(header + "\n")
            for hs in self.hsm.hotspots:
                newrowtxt = hs.toCSVRow()
                temp_file.write(newrowtxt)
