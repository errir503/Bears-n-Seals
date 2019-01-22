import random

from arcticapi.augmnetation.TrainingChip import TrainingChip
from arcticapi.model.HotSpot import SpeciesList

from utils import *
import numpy as np

def prepare_chips(cfg, aeral_image, bounding_boxes):
    """
    :param cfg: CropCfg
    :type hs: HotSpot
    """
    if not aeral_image.file_exists:
        return []

    w, h = aeral_image.w, aeral_image.h

    chips = []
    drawn = []

    for bbox in bounding_boxes:
        # if already drawn skip
        found = False
        for hsId in drawn:
            if bbox.hsId == hsId:
                found = True
        if found:
            continue

        tcrop, bcrop, lcrop, rcrop, center_x, center_y, dx, dy = recalculate_crops(bbox.y2, bbox.y1, bbox.x1, bbox.x2,
                                                                                   h, w, cfg.maxShift,
                                                                                   cfg.minShift,
                                                                                   cfg.crop_size)
        # create crop
        crop_img = np.zeros([bcrop-tcrop, rcrop-lcrop, 3], dtype=np.uint8)


        # shift bounding boxes that fit the new crop dimensions
        shifted_bboxs = []
        for bb in bounding_boxes:
            bbs_shifted = bb.shift(left=-lcrop, top=-tcrop)
            bbs_shifted.hsId = bb.hsId
            shifted_bboxs.append(bbs_shifted)

        if not bbox.shift(left=-lcrop, top=-tcrop).is_fully_within_image(crop_img):
            print("For an odd reason hotspot " + bbox.hsId + " did not fully fit in the new box:(%d, %d)(%d, %d) crop: (%d, %d)(%d, %d)" %
                  (bbox.x1, bbox.y2, bbox.x2, bbox.y1, lcrop, bcrop, rcrop, tcrop))

        # check if is within image, only add to drawn if is fully in image
        to_draw = []
        for bb in shifted_bboxs:
            if bb.is_partly_within_image(crop_img):
                new = bb.cut_out_of_image(crop_img)
                if new.area < bb.area * 0.5:
                    print("over 1/2 of bbox cut from %s so skipping" % bb.hsId)
                    continue
                new.hsId = bb.hsId
                to_draw.append(new)

            if bb.is_fully_within_image(crop_img):
                drawn.append(bb.hsId)


        if len(to_draw) == 0:
            print("0 bboxes")
            continue
        tr = TrainingChip(aeral_image, crop_img.shape, cfg, aeral_image.path, to_draw, (tcrop, bcrop, lcrop, rcrop))
        del crop_img

        chips.append(tr)
        for bbox in bounding_boxes:
            contains = False
            for drawnid in drawn:
                if bbox.hsId == drawnid:
                    contains = True
            if not contains:
                print("Did not draw " + bbox.hsId)

    uniquechips = []
    for idx, chip in enumerate(chips):
        found = False
        for uchip in uniquechips:
            if chip.filename == uchip.filename:
                found = True
        if not found:
            uniquechips.append(chip)
        else:
            chip.filename = chip.filename + "-" + str(idx)
            uniquechips.append(chip)

    return uniquechips

def test_train_split(chips):
    random.shuffle(chips)
    x = int(len(chips) / 4) * 3  # 3/4 train 1/4 test
    train = chips[:x]
    test = chips[x:]
    return train, test

def equalize_classes(chips):
    dict_area = class_chip_dict_area(chips)
    dict = class_chip_dict(chips)
    counts = []
    for k in dict_area:
        vals = dict_area[k]
        counts.append(len(vals))

    max = np.max(counts)

    imgs_to_add = np.subtract(max, counts)

    for idx, ct in enumerate(imgs_to_add):
        # We only care about ringed/bearded don't really care about UNK seals
        if not idx == 0 and not idx == 1:
            continue

        class_chips = dict[idx]
        class_len = len(class_chips)
        added = 0
        i = 0
        x_copies = 0
        while added < ct:
            if i % class_len == 0:
                x_copies += 1
            # allow up to 3x duplicates
            if x_copies > 4:
                break
            copy = class_chips[i % class_len].copy()
            copy.filename = copy.filename + "x" + str(x_copies)
            dict[idx].append(copy)
            added += len(copy.bboxes.bounding_boxes)
            i += 1
        print("Added %d %ss" % (added, SpeciesList[idx]))

    new_chips =  sum(dict.values(), [])
    return new_chips

def print_bbox_stats(chips):
    dict = class_chip_dict_area(chips)
    for k in dict:
        vals = dict[k]

        arr = np.asarray(vals)
        avg = np.mean(arr)
        stddev = np.std(arr, ddof=1)
        if np.isnan(stddev):
            stddev = -1
        print("Class:%s Count:%d Avg_area:%d Stddev_area:%d" % (
        SpeciesList[k], len(vals), int(avg), int(stddev)))


def class_chip_dict_area(chips):
    boxes = [b for c in chips for b in c.bboxes.bounding_boxes]
    dict = {}
    for box in boxes:
        if box.label not in dict:
            dict[box.label] = []

        dict[box.label].append(box.area)
    return dict


def class_chip_dict(chips):
    dict = {}
    for chip in chips:
        for box in chip.bboxes.bounding_boxes:
            if box.label not in dict:
                dict[box.label] = []
            dict[box.label].append(chip)
            break

    return dict