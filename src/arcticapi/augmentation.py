import os
import cv2
import arcticapi.normalizer as norm
import numpy as np
from random import randint

# Config object for cropping/augmentation parameters
from arcticapi.visuals import plot_px_distribution


class CropCfg(object):
    def __init__(self, csv, im_dir, out_dir, bbox_size, minShift, maxShift, crop_size, label, combine_seal, make_bear, make_anomaly,
                 debug, imtype, name, combine_all):
        self.csv = csv
        self.im_dir = im_dir
        self.out_dir = out_dir
        self.bbox_size = bbox_size
        self.minShift = minShift
        self.maxShift = maxShift
        self.crop_size = crop_size
        self.label = label
        self.combine_seal = combine_seal
        self.make_bear = make_bear
        self.make_anomaly = make_anomaly
        self.combine_all = combine_all
        self.debug = debug
        self.imtype = imtype
        self.name = name

def crop_ir_hotspot_8bit(cfg, hs):
    """
    :param cfg: CropCfg
    :type hs: HotSpot
    """
    if not hs.ir.load_image():
        return

    base_name = os.path.basename(hs.ir.path)
    file_name = cfg.out_dir + os.path.splitext(base_name)[0]
    img = hs.ir.image
    img = img.astype(np.uint16)
    # img = np.array(img).astype(np.float32)
    # img = np.square(img)
    img = norm.lin_normalize_image(img, True)

    # plot_px_distribution(imgpre, img, "POST NORM DISTRIBUTION", 10000)

    if cfg.debug:
        crop_path = file_name + ".jpg"
        if os.path.isfile(crop_path):
            hs.ir.free()
            img = cv2.imread(crop_path)

    classIndex = hs.classIndex
    id = hs.id
    imgh = img.shape[0]
    imgw = img.shape[1]

    center_x = hs.thermal_loc[0]
    center_y = hs.thermal_loc[1]
    if center_y == 0 or center_x == 0:
        print("Cant parse id " + id + " because center x or y is 0 which is invalid darknet lablel")
        return
    if cfg.debug:
        # cv2.circle(img, hs.thermal_loc, 5, (0, 255, 0), 2)
        cv2.rectangle(img, (center_x - cfg.bbox_size / 2, center_y - cfg.bbox_size / 2),
                      (center_x + cfg.bbox_size / 2, center_y + cfg.bbox_size / 2),
                      (0, 255, 0), 1)  # draw rect

    cv2.imwrite(file_name + ".jpg", img)

    fileExisted = os.path.isfile(file_name + ".txt")
    # Generate trainin label
    with open(file_name + ".txt", 'a') as file:
        file.write(str(classIndex) + " " + str((center_x + 0.0) / imgw) + " " +
                   str((center_y + 0.0) / imgh) + " " +
                   str((cfg.bbox_size + 0.0) / imgw) + " " +
                   str((cfg.bbox_size + 0.0) / imgh) + "\n")

    # if file doesnt already exist
    if not fileExisted:
        with open(cfg.label, 'a') as file:
            file.write(os.getcwd() + "/" + file_name + ".jpg" + "\n")

    # free image from memory
    hs.rgb.free()

# Given a 16-bit 1-channel image saves at .tif
def crop_ir_hotspot_16bit(cfg, hs):
    """
    :param cfg: CropCfg
    :type hs: HotSpot
    """
    if not hs.ir.load_image():
        return

    base_name = os.path.basename(hs.ir.path)
    file_name = cfg.out_dir + os.path.splitext(base_name)[0]
    img = hs.ir.image

    if cfg.debug:
        crop_path = file_name + ".tif"
        if os.path.isfile(crop_path):
            hs.ir.free()
            img = cv2.imread(crop_path)

    classIndex = hs.classIndex
    id = hs.id
    imgh = img.shape[0]
    imgw = img.shape[1]

    center_x = hs.thermal_loc[0]
    center_y = hs.thermal_loc[1]
    if center_y == 0 or center_x == 0:
        print("Cant parse id " + id + " because center x or y is 0 which is invalid darknet lablel")
        return
    if cfg.debug:
        # cv2.circle(img, hs.thermal_loc, 5, (0, 255, 0), 2)
        cv2.rectangle(img, (center_x - cfg.bbox_size / 2, center_y - cfg.bbox_size / 2),
                      (center_x + cfg.bbox_size / 2, center_y + cfg.bbox_size / 2),
                      (0, 255, 0), 1)  # draw rect

    cv2.imwrite(file_name + ".tif", img)

    fileExisted = os.path.isfile(file_name + ".txt")
    # Generate trainin label
    with open(file_name + ".txt", 'a') as file:
        file.write(str(classIndex) + " " + str((center_x + 0.0) / imgw) + " " +
                   str((center_y + 0.0) / imgh) + " " +
                   str((cfg.bbox_size + 0.0) / imgw) + " " +
                   str((cfg.bbox_size + 0.0) / imgh) + "\n")

    # if file doesnt already exist
    if not fileExisted:
        with open(cfg.label, 'a') as file:
            file.write(os.getcwd() + "/" + file_name + ".tif" + "\n")

    # free image from memory
    hs.rgb.free()

def crop_rgb_hotspot(cfg, hs):
    """
    :param cfg: CropCfg
    :type hs: HotSpot
    """
    if not hs.rgb.load_image():
        return

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
    crop_img = brightness_adjustment(crop_img)
    if cfg.debug:
        cv2.circle(crop_img, (center_x, center_y), 5, (0, 255, 0), 2)
        cv2.rectangle(crop_img, (center_x - cfg.bbox_size /2, center_y - cfg.bbox_size /2),
                      (center_x + cfg.bbox_size /2, center_y + cfg.bbox_size /2),
                      (0, 255, 0), 2)  # draw rect

    croph = crop_img.shape[0]
    cropw = crop_img.shape[1]
    file_name = cfg.out_dir + "crop_" + id + "_" + str(classIndex)
    cv2.imwrite(file_name + ".jpg", crop_img)

    # Generate trainin label
    with open(file_name + ".txt", 'a') as file:
        file.write(str(classIndex) + " " + str((center_x + 0.0) / cropw) + " " +
                   str((center_y + 0.0) / croph) + " " +
                   str((cfg.bbox_size + 0.0) / cropw) + " " +
                   str((cfg.bbox_size + 0.0) / croph) + "\n")

    write_label(file_name, cfg.label)

    if False: #TODO temporary
        # Generate negative image(with no object) and labels for training
        tcrop, bcrop, lcrop, rcrop = negative_bounds(tcrop, bcrop, lcrop, rcrop, imgw, imgh, cfg.crop_size)
        crop_img_neg = img[tcrop:bcrop, lcrop: rcrop]
        file_name = cfg.out_dir + "crop_" + id + "_" + str(classIndex)
        cv2.imwrite(file_name + ".jpg", crop_img)

        file_name_neg = cfg.out_dir + "crop_" + id + "_" + str(classIndex) + "_neg"
        open(file_name_neg + ".txt", 'a').close()
        cv2.imwrite(file_name_neg + ".jpg", crop_img_neg)
        write_label(file_name_neg, cfg.label)

    # free image from memory
    hs.rgb.free()


def write_label(file_name, label_files_list):
    with open(label_files_list, 'a') as file:
        file.write(os.getcwd() + "/" + file_name + ".jpg" + "\n")

def recalculate_crops(rgb_bb_b, rgb_bb_t, rgb_bb_l, rgb_bb_r, imgh, imgw, maxShift, minShift, crop_size):
    # center points of bounding box in the image
    center_y_global = rgb_bb_t + (rgb_bb_b - rgb_bb_t) / 2
    center_x_global = rgb_bb_l + (rgb_bb_r - rgb_bb_l) / 2

    lcrop_orig = center_x_global - crop_size/2
    rcrop_orig = center_x_global + crop_size/2
    tcrop_orig = center_y_global - crop_size/2
    bcrop_orig = center_y_global + crop_size/2

    dx, dy = random_shift(tcrop_orig, bcrop_orig, lcrop_orig, rcrop_orig, imgw, imgh, minShift, maxShift)

    lcrop = lcrop_orig + dx
    rcrop = rcrop_orig + dx
    bcrop = bcrop_orig + dy
    tcrop = tcrop_orig + dy

    # Ensure hotspot is still in cropped space, if not shift so that it is
    if center_x_global < lcrop:
        diff = lcrop - center_x_global
        lcrop -= diff
        rcrop -= diff

    if center_x_global > rcrop:
        diff = center_x_global - rcrop
        lcrop += diff
        rcrop += diff

    if center_y_global < tcrop:
        diff = center_y_global - tcrop
        bcrop += diff
        tcrop += diff

    if center_y_global > bcrop:
        diff = bcrop - center_y_global
        bcrop -= diff
        tcrop -= diff

    if tcrop < 0:
        diff = 0 - tcrop
        tcrop += diff
        bcrop += diff
    if bcrop > imgh:
        diff = bcrop - imgh
        bcrop -= diff
        tcrop -= diff
    if lcrop < 0:
        diff = 0 - lcrop
        lcrop += diff
        rcrop += diff
    if rcrop > imgw:
        diff = imgw - rcrop
        rcrop -= diff
        lcrop -= diff



    dx = lcrop_orig - lcrop
    dy = tcrop_orig - tcrop
    local_x = crop_size/2 + dx
    local_y = crop_size/2 + dy
    return tcrop, bcrop, lcrop, rcrop, local_x, local_y


def negative_bounds(topCrop, bottomCrop, leftCrop, rightCrop, w, h, crop_size):
    # TODO find points screen quadrant take from another quadrant
    offset = 50
    mid_y = (bottomCrop - topCrop)/2 + topCrop
    mid_x = (rightCrop - leftCrop)/2 + leftCrop

    if mid_x > 800 or mid_y > 800 :
        # bottom left corner
        return h-crop_size, h, 0, crop_size
    else:
        # top right corner
        return 0, crop_size, h-crop_size, h




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

def brightness_adjustment(img):
    # turn the image into the HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # creates a random bright
    ratio = .7 + np.random.uniform()
    # convert to int32, so you don't get uint8 overflow
    # multiply the HSV Value channel by the ratio
    # clips the result between 0 and 255
    # convert again to uint8
    hsv[:,:,2] =  np.clip(hsv[:,:,2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
    # return the image int the BGR color space
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)