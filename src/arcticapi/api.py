import os
import csv
import random
import sys

from augmnetation import AugRgb
from augmnetation.utils import write_label
from csv_parser import parse_hotspot
from registration import image_registration
from model.HotSpotMap import HotSpotMap
from visuals import print_loading_bar


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
        rgb_im = {}
        ir_im = {}
        for row in rows:
            hotspot = parse_hotspot(row, im_path)
            hsm.add(hotspot)
            if not hotspot.rgb.path in rgb_im:
                rgb_im[hotspot.rgb.path] = hotspot.rgb
            if not hotspot.ir.path in ir_im:
                ir_im[hotspot.ir.path] = hotspot.ir
            rgb_im[hotspot.rgb.path].hotspots.append(hotspot)
            ir_im[hotspot.ir.path].hotspots.append(hotspot)


        self.rgb_images = rgb_im
        self.ir_images = ir_im
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

    def generate_training_set(self, cfg):
        img_ct = len(self.rgb_images)
        print("processing " + str(img_ct) + " images")
        if not os.path.exists(cfg.out_dir):
            os.mkdir(cfg.out_dir)

        label_base = cfg.label.split(".")[0]
        chips = []
        for image_path in self.rgb_images:
            chips = chips + self.rgb_images[image_path].generate_chips(cfg)

        print("\nOriginal Stats:")
        AugRgb.print_bbox_stats(chips)
        print


        if cfg.combine_seal:
            for chip in chips:
                for bbs in chip.bboxes.bounding_boxes:
                    if (bbs.label == 0 or bbs.label == 1 or bbs.label == 2):
                        bbs.label = 0

        for chip in chips:
            for bbs in chip.bboxes.bounding_boxes:
                if bbs.label == 2:
                    bbs.label = 0
        # Test/Train split
        train, test = AugRgb.test_train_split(chips)
        # train = AugRgb.equalize_classes(train)
        print("Attempting to make classes of similar size")
        # train = AugRgb.equalize_classes(train)
        print
        AugRgb.print_bbox_stats(train)
        AugRgb.print_bbox_stats(test)

        print("Training set stats:")
        print("Generating training set...")
        for idx, c in enumerate(train):
            pct = ((idx + 0.0) / len(train)) * 100.0
            sys.stdout.write("\r|%-73s| %3d%%" % ('#' * int(pct * .73), pct))
            # models tend to struggle with larger seals so allow more zoom in than out
            # if not c.load(random.uniform(-.05, 0.1)):
            if not c.load():
                print("Chip not loaded in api.py :( %s" % c.filename)
                continue
            # augmentations`
            c.color_change(-5, 5, False)
            # c.extend(2)
            # c.flip()
            # c.rotate()
            c.save()  # save image
            write_label(c.filename + ".jpg", label_base + "_train.txt")
            c.aeral_image.free()
            c.free()

        print("Testing set stats:")
        print("Generating test set...")
        for idx, chip in enumerate(test):
            print_loading_bar(((idx + 0.0) / len(test)) * 100.0)
            if not chip.load():
                print("Chip not loaded in api.py :(")
                continue
            chip.extend(2)
            chip.save()  # save image and labels
            chip.aeral_image.free()
            chip.free()  # free image
            write_label(chip.filename + ".jpg", label_base + "_test.txt")

        print("COMPLETE")
        return

    def addHotspot(self, hs):
        self.rgb_images[hs.rgb.path].hotspots.append(hs)
        self.hsm.add(hs)

    def setStatus(self, hs, status):
        for hotspot in self.rgb_images[hs.rgb.path].hotspots:
            if hotspot.id == hs.id:
                hs.status = status
        self.hsm.get_hs(hs.id).status = status

    def updateHs(self, hs, updated):
        for hotspot in self.rgb_images[hs.rgb.path].hotspots:
            if hotspot.id == hs.id:
                hs.updated = updated
        self.hsm.get_hs(hs.id).updated = updated

    # Save all hotspots unfiltered in the standard seal csv format
    def saveHotspotsToCSV(self, out_file, header):
        with open(out_file, 'w') as temp_file:
            temp_file.write(header + "\n")
            for hs in self.hsm.hotspots:
                newrowtxt = hs.toCSVRow()
                temp_file.write(newrowtxt)
