import os
import cv2
import numpy as np
from src.arcticapi import normalizer


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
    img = normalizer.lin_normalize_image(img, True)

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