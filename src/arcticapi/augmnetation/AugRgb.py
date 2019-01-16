import imgaug as ia

from arcticapi.augmnetation.TrainingChip import TrainingChip
from utils import *

def prepare_chips(cfg, aeral_image):
    """
    :param cfg: CropCfg
    :type hs: HotSpot
    """
    if not aeral_image.load_image():
        return

    bboxes = aeral_image.getBboxes(cfg)
    img = aeral_image.image

    chips = []
    drawn = []
    for bbox in bboxes.bounding_boxes:
        # already drawn skip
        found = False
        for hsId in drawn:
            if bbox.hsId == hsId:
                found = True
        if found:
            continue

        tcrop, bcrop, lcrop, rcrop, center_x, center_y, dx, dy = recalculate_crops(bbox.y2, bbox.y1, bbox.x1, bbox.x2,
                                                                                   img.shape[0], img.shape[1], cfg.maxShift,
                                                                                   cfg.minShift,
                                                                                   cfg.crop_size)

        # create crop
        crop_img = img[tcrop:bcrop, lcrop: rcrop]

        # shift bounding boxes that fit the new crop dimensions
        shifted_bboxs = []
        for bb in bboxes.bounding_boxes:
            bbs_shifted = bb.shift(left=-lcrop, top=-tcrop)
            bbs_shifted.hsId = bb.hsId
            shifted_bboxs.append(bbs_shifted)

        if not bbox.shift(left=-lcrop, top=-tcrop).is_fully_within_image(crop_img):
            print("WAAA")

        # check if
        to_draw = []
        for bb in shifted_bboxs:
            # TODO is fully or is_partly_within_image?
            if bb.is_partly_within_image(crop_img):
                to_draw.append(bb)
            if bb.is_fully_within_image(crop_img):
                drawn.append(bb.hsId)

        tr = TrainingChip(crop_img, cfg, aeral_image.path, to_draw, (tcrop, bcrop, lcrop, rcrop))
        tr.extend(5)
        chips.append(tr)

    # free image from memory
    aeral_image.free()
    for bbox in bboxes.bounding_boxes:
        contains = False
        for drawnid in drawn:
            if bbox.hsId == drawnid:
                contains = True

        if not contains:
            print("Did not draw " + bbox.hsId) # todo 78376

    for chip in chips:
        chip.save()




