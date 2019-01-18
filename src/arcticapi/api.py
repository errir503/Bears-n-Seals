import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from arcticapi.augmnetation.utils import write_label
from arcticapi.csv_parser import parse_hotspot
from arcticapi.model.HotSpot import SpeciesList
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
            chips = chips + self.images[image_path].generate_chips(cfg)

        all_bboxes = [b for c in chips for b in c.bboxes.bounding_boxes ]
        self.print_bbox_stats(all_bboxes)

        random.shuffle(chips)
        x = int(len(chips)/4) * 3  # 3/4 train 1/4 test
        train = chips[:x]
        test = chips[x:]
        print("Chipping complete %d chips created" % len(chips))
        print("Starting data augmentation, cropping, and label generation")
        for chip in train:
            if not chip.load():
                print("Chip not loaded in api.py :( %s" % chip.filename)
                continue
            copy = chip.copy()
            copy.load()
            copy.filename = copy.filename + "_b"
            chips = [chip, copy]
            for c in chips:
                if c.image is None:
                    print "Skipped " + c.imgpath
                    continue
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
            if not chip.load():
                print("Chip not loaded in api.py :(")
                continue
            chip.load() # load image
            chip.extend(5)
            chip.save() # save image and labels
            chip.free() # free image
            write_label(chip.filename + ".jpg", label_base + "_test.txt")

    def print_bbox_stats(self, boxes):
        dict = {}
        for box in boxes:
            if box.label not in dict:
                dict[box.label] = []

            dict[box.label].append(box.area)

        for k in dict:
            vals = dict[k]
            # plt.title(SpeciesList[k])
            # plt.hist(vals, fc='red', rwidth=1, bins=20)
            # plt.show()

            arr = np.asarray(vals)
            avg = np.mean(arr)
            stddev = np.std(arr, ddof=1)
            if np.isnan(stddev):
                stddev = -1
            print("Class:%s Count:%d Avg_area:%d Stddev_area:%d" % (SpeciesList[k], len(vals), int(avg), int(stddev)))



    # Save all hotspots unfiltered in the standard seal csv format
    def saveHotspotsToCSV(self, out_file):
        with open(out_file, 'w') as temp_file:
            temp_file.write(",".join(self.csvheader) + "\n")
            for hs in self.hsm.hotspots:
                newrowtxt = hs.toCSVRow()
                temp_file.write(newrowtxt)


