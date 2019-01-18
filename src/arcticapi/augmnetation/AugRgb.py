from arcticapi.augmnetation.TrainingChip import TrainingChip
from utils import *
import numpy as np

def prepare_chips(cfg, aeral_image):
    """
    :param cfg: CropCfg
    :type hs: HotSpot
    """
    if not aeral_image.file_exists:
        return []

    bboxes = aeral_image.getBboxes(cfg)
    w, h = aeral_image.w, aeral_image.h

    chips = []
    drawn = []
    for bbox in bboxes:
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
        for bb in bboxes:
            bbs_shifted = bb.shift(left=-lcrop, top=-tcrop)
            bbs_shifted.hsId = bb.hsId
            shifted_bboxs.append(bbs_shifted)

        if not bbox.shift(left=-lcrop, top=-tcrop).is_fully_within_image(crop_img):
            print("For an odd reason hotspot " + bbox.hsId + " did not fully fit in the crop")

        # check if is within image, only add to drawn if is fully in image
        to_draw = []
        for bb in shifted_bboxs:
            if bb.is_partly_within_image(crop_img):
                new = bb.cut_out_of_image(crop_img)
                if new.area < bbox.area * 0.5:
                    print("TOO MUCH REMOVED FROM %s" % bb.hsId)
                    continue
                new.hsId = bbox.hsId
                to_draw.append(new)

            if bb.is_fully_within_image(crop_img):
                drawn.append(bb.hsId)


        if len(to_draw) == 0:
            print("0 bboxes")
            continue
        tr = TrainingChip(aeral_image, crop_img.shape, cfg, aeral_image.path, to_draw, (tcrop, bcrop, lcrop, rcrop))
        del crop_img

        chips.append(tr)
    # free image from memory
    # aeral_image.free() TODO remove
    for bbox in bboxes:
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




