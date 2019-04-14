import glob
import os
import csv
import random
import sys

from augmnetation import AugRgb
from augmnetation.utils import write_label
from csv_parser import parse_hotspot, parse_hotspot_new_dataset
from registration import image_registration
from model.HotSpotMap import HotSpotMap
from model.AerialImage import AerialImage
from visuals import print_loading_bar, plot_sizes


# This is the controller of the api which is created using the path of the full resolution image directory
# and the path to the NOAA hotspot csv file.
# Once created contains a list of hotspots and a list of AerialImages which contain all information about the image,
# and hotspots within that image
class ArcticApi:
    def __init__(self, cfg):
        rows = list()

        f = open(cfg.csv, 'r')
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
        f.close()
        del rows[0]  # remove col headers

        hsm = HotSpotMap()
        rgb_im = {}
        ir_im = {}
        self.NA_rows = []
        for row in rows:
            hotspot = parse_hotspot_new_dataset(row, cfg)
            if hotspot is None:
                self.NA_rows.append(row)
                continue

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
        self.cfg = cfg
        del rows

    def register(self, id=None, showFigures=False, showImgs=False):
        if id is None:
            for hs in self.hsm.hotspots:
                image_registration.register_images(hs, showFigures, showImgs)
        else:
            hs = self.hsm.get_hs(id)
            if hs is not None:
                image_registration.register_images(hs, showFigures, showImgs)
    def generate_training_ir(self, cfg):
        background_path = "/data/raw_data/VIAME_BACKGROND_NORMALIZED/*.PNG"
        background_images = []
        all_im  = glob.glob(background_path)
        all_im_len = len(all_im)
        for idx, filename in enumerate(all_im):  # assuming gif
            ir = AerialImage(filename, "ir", None)
            chip = ir.generate_chips(cfg)
            new_name = "background_%d" % idx
            chip.filename = cfg.out_dir + new_name
            path = chip.save()  # save image and labels
            background_images.append(path)
            print_loading_bar(((idx + 0.0) / all_im_len) * 100.0)

            chip.aeral_image.free()

        hotspot_images = []
        all_im_len = len(self.ir_images)
        for idx, image_path in enumerate(self.ir_images):
            chip = self.ir_images[image_path].generate_chips(cfg)
            path = chip.save()  # save image and labels
            hotspot_images.append(path)
            chip.aeral_image.free()
            print_loading_bar(((idx + 0.0) / all_im_len) * 100.0)



    def generate_training_set(self, cfg):
        img_ct = len(self.rgb_images)
        print("processing " + str(img_ct) + " images")
        if not os.path.exists(cfg.out_dir):
            os.mkdir(cfg.out_dir)

        label_base = cfg.label.split(".")[0]
        chips = []
        path = ""
        max = 0
        for image_path in self.rgb_images:
            if len(self.rgb_images[image_path].hotspots) > max:
                max = len(self.rgb_images[image_path].hotspots)
                path = self.rgb_images[image_path].path
            chips = chips + self.rgb_images[image_path].generate_chips(cfg)
        good_chips = []
        for chip in chips:
            is_ok = True
            for bb in chip.bboxes.bounding_boxes:
                hs = self.hsm.get_hs(bb.hsId)
                if hs.classIndex > 2:
                    is_ok = False
                if "bad_res" in hs.status:
                    is_ok = False
                if not hs.updated:
                    is_ok = False
                if hs.isStatusRemoved():
                    is_ok = False
            if is_ok:
                good_chips.append(chip)
        chips = good_chips

        plot_sizes(chips, self)
        print("\nOriginal Stats:")
        AugRgb.print_bbox_stats(chips)
        print


        if cfg.combine_seal:
            for chip in chips:
                for bbs in chip.bboxes.bounding_boxes:
                    if (bbs.label == 0 or bbs.label == 1 or bbs.label == 2):
                        bbs.label = 0
                    if (bbs.label == 2):
                        bbs.label = 0
        for chip in chips:
            for bbs in chip.bboxes.bounding_boxes:
                if (bbs.label == 2):
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
            # c.color_change(-5, 5, False)
            # c.extend(2)
            # c.flip()
            c.extend(2)
            c.save()  # save image
            write_label(c.filename + ".jpg", cfg.out_dir+label_base + "_train.txt")
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
            write_label(chip.filename + ".jpg", cfg.out_dir+label_base + "_test.txt")

        print("COMPLETE")
        return

    def addHotspot(self, hs):
        self.rgb_images[hs.rgb.path].hotspots.append(hs)
        self.hsm.add(hs)

    def setStatus(self, hs, status):
        hs.status = status
        for hotspot in self.rgb_images[hs.rgb.path].hotspots:
            if hotspot.id == hs.id:
                hotspot.status = hs.status
        self.hsm.get_hs(hs.id).status = hs.status

    def setType(self, hs, type):
        hs.type = type
        for hotspot in self.rgb_images[hs.rgb.path].hotspots:
            if hotspot.id == hs.id:
                hotspot.type = hs.type
        self.hsm.get_hs(hs.id).type = hs.type

    def setClass(self, hs):
        for hotspot in self.rgb_images[hs.rgb.path].hotspots:
            if hotspot.id == hs.id:
                hotspot.classIndex = hs.classIndex
                hotspot.species = hs.species
                hotspot.rgb_bb.label = hs.classIndex

        self.hsm.get_hs(hs.id).classIndex = hs.classIndex
        self.hsm.get_hs(hs.id).species = hs.species
        self.hsm.get_hs(hs.id).rgb_bb.label = hs.classIndex

    def updateHs(self, hs, updated):
        hs.updated = updated
        for hotspot in self.rgb_images[hs.rgb.path].hotspots:
            if hotspot.id == hs.id:
                hotspot.updated = hs.updated
        self.hsm.get_hs(hs.id).updated = hs.updated

    def getImagesWithSeals(self, ir = False):
        all_images = self.rgb_images.keys()
        if ir:
            all_images = self.ir_images.keys()
        images_w_seals = []
        for img in all_images:
            hasSeal = False
            images = self.ir_images if ir else self.rgb_images
            for hs in images[img].hotspots:
                if hs.species in ["Ringed Seal", "Bearded Seal", "UNK Seal"]:
                    hasSeal = True
            if hasSeal:
                images_w_seals.append(img)
        return images_w_seals


    # Save all hotspots unfiltered in the standard seal csv format
    def saveHotspotsToCSV(self, out_file, header):
            self.saveHotspots(self.hsm.hotspots, out_file, header, True)

    def saveHotspots(self, hotspots, out_file, header, NA_ROW = False):
        with open(out_file, 'w') as temp_file:
            temp_file.write(header + "\n")
            for hs in hotspots:
                newrowtxt = hs.toCSVRow(True)
                temp_file.write(newrowtxt)
            if NA_ROW:
                for na_row in self.NA_rows:
                    temp_file.write(','.join(na_row) + '\n')
