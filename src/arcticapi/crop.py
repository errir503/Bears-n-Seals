import os
import cv2
from random import randint


class CropCfg(object):
    def __init__(self, out_dir, bbox_size, minShift, maxShift, crop_size, label, combine_seal, make_bear, make_anomaly,
                 debug=False):
        self.out_dir = out_dir
        self.bbox_size = bbox_size
        self.minShift = minShift
        self.maxShift = maxShift
        self.crop_size = crop_size
        self.label = label
        self.combine_seal = combine_seal
        self.make_bear = make_bear
        self.make_anomaly = make_anomaly
        self.debug = debug

    def tostr(self):
        return ("combine_seal = " + str(self.combine_seal) + "\n" +
                "Generating polar bear crops = " + str(self.make_bear) + "\n" +
                "generating anomaly crops crops = " + str(self.make_anomaly) + "\n\n" +
                "bounding box w/h: " + str(self.bbox_size) + "\n" +
                "crop size: " + str(self.crop_size) + "\n" +
                "output to crops and labels saving to: " + self.out_dir + "\n" +
                "training label list saving to: " + self.label)


def crop_hotspot(cfg, hs):
    """

    :param cfg: CropCfg
    :type hs: HotSpot
    """
    img = hs.rgb.image
    classIndex = hs.classIndex
    id = hs.id
    imgh = img.shape[0]
    imgw = img.shape[1]

    tcrop, bcrop, lcrop, rcrop, center_x, center_y = recalculate_crops(hs.rgb_bb_b, hs.rgb_bb_t, hs.rgb_bb_l,
                                                                       hs.rgb_bb_r,
                                                                       imgh, imgw, cfg.maxShift, cfg.minShift,
                                                                       cfg.crop_size)

    crop_img = img[tcrop:bcrop, lcrop: rcrop]

    if cfg.debug:
        cv2.circle(crop_img, (center_x, center_y), 5, (0, 255, 0), 2)
        cv2.rectangle(crop_img, (center_x - cfg.bbox_size, center_y - cfg.bbox_size),
                      (center_x + cfg.bbox_size, center_y + cfg.bbox_size),
                      (0, 255, 0), 2)  # draw rect

    croph = crop_img.shape[0]
    cropw = crop_img.shape[1]
    file_name = cfg.out_dir + "crop_" + id + "_" + str(classIndex)
    cv2.imwrite(file_name + ".jpg", crop_img)

    tcropn, bcropn, lcropn, rcropn = negative_bounds(tcrop, bcrop, lcrop, rcrop, imgw, imgh)
    crop_img_neg = img[tcropn:bcropn, lcropn: rcropn]
    file_name = cfg.out_dir + "crop_" + id + "_" + str(classIndex)
    cv2.imwrite(file_name + ".jpg", crop_img)

    # Generate negative image for training
    file_name_neg = cfg.out_dir + "crop_" + id + "_" + str(classIndex) + "_neg"
    open(file_name_neg + ".txt", 'a').close()
    cv2.imwrite(file_name_neg + ".jpg", crop_img_neg)

    with open(file_name + ".txt", 'a') as file:
        file.write(str(classIndex) + " " + str((center_x + 0.0) / cropw) + " " +
                   str((center_y + 0.0) / croph) + " " +
                   str((cfg.bbox_size + 0.0) / cropw) + " " +
                   str((cfg.bbox_size + 0.0) / croph) + "\n")

    with open(cfg.label, 'a') as file:
        file.write(os.getcwd() + "/" + file_name + ".jpg" + "\n")
        file.write(os.getcwd() + "/" + file_name_neg + ".jpg" + "\n")


def recalculate_crops(rgb_bb_b, rgb_bb_t, rgb_bb_l, rgb_bb_r, imgh, imgw, maxShift, minShift, crop_size):
    # center points of bounding box in the image
    center_y_global = rgb_bb_t + (rgb_bb_b - rgb_bb_t) / 2
    center_x_global = rgb_bb_l + (rgb_bb_r - rgb_bb_l) / 2

    # recalculate crops
    rgb_bb_b = center_y_global + crop_size / 2
    rgb_bb_t = center_y_global - crop_size / 2
    rgb_bb_l = center_x_global - crop_size / 2
    rgb_bb_r = center_x_global + crop_size / 2

    center_y = (rgb_bb_b - rgb_bb_t) / 2
    center_x = (rgb_bb_r - rgb_bb_l) / 2

    tcrop = max(rgb_bb_t, 0)
    bcrop = max(rgb_bb_b, 0)
    bcrop = min(bcrop, imgh)
    lcrop = max(rgb_bb_l, 0)
    rcrop = max(rgb_bb_r, 0)
    rcrop = min(rcrop, imgw)

    dx, dy = random_shift(tcrop, bcrop, lcrop, rcrop, imgw, imgh, minShift, maxShift)

    lcrop += dx
    rcrop += dx
    bcrop += dy
    tcrop += dy
    center_y -= dy
    center_x -= dx

    tcrop = max(tcrop, 0)
    bcrop = max(bcrop, 0)
    bcrop = min(bcrop, imgh)
    lcrop = max(lcrop, 0)
    rcrop = max(rcrop, 0)
    rcrop = min(rcrop, imgw)

    if tcrop == 0:
        center_y += rgb_bb_t
    if lcrop == 0:
        center_x += rgb_bb_l

    return tcrop, bcrop, lcrop, rcrop, center_x, center_y


def negative_bounds(topCrop, bottomCrop, leftCrop, rightCrop, w, h):
    offset = 50
    if leftCrop > w / 2:
        # take negative sample from right half
        return topCrop + offset, bottomCrop + offset, w - 512, w
    elif rightCrop < w / 2:
        # take negative sample from left half
        return topCrop + offset, bottomCrop + offset, 0, 512

    # elif topCrop < h / 2:
    #     # take negative sample from left half
    #     return bottomCrop, bottomCrop+512, leftCrop, rightCrop

    return 0, 512, 0, 512


def random_shift(topCrop, bottomCrop, leftCrop, rightCrop, w, h, minShift, maxShift):
    if maxShift == 0:
        return 0, 0

    dx = 0
    dy = 0

    # make dx
    if randint(0, 1) == 1:
        if leftCrop != 0 and randint(0, 1) == 1:
            dx -= randint(minShift, maxShift)

        elif rightCrop != 0 and randint(0, 1) == 1:
            dx += randint(minShift, maxShift)

        # left crop outside image bounds
        if not leftCrop + dx > 0:
            dx = 0
        if not rightCrop + dx < w:
            dx = 0

    # make dy
    if randint(0, 1) == 1:
        if topCrop != 0 and randint(0, 1) == 1:
            dy -= randint(minShift, maxShift)

        elif bottomCrop != 0 and randint(0, 1) == 1:
            dy += randint(minShift, maxShift)

        # left crop outside iomage bounds
        if topCrop + dy < 0:
            dy = 0
        if bottomCrop + dx > h:
            dy = 0

    return dx, dy
