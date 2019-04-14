import math
import os
import cv2
import numpy as np
from src.arcticapi import normalizer
import TrainingChip
from src.arcticapi.model.BoundingBox import BoundingBox
import time

timerc = 0
totalc = 0
def crop_ir_hotspot_8bit(cfg, aerial_image):
    """
    :param cfg: CropCfg
    :type aerial_image: HotSpot
    """
    if not aerial_image.load_image():
        return

    img = aerial_image.image
    # start = time.time()
    # mi = np.percentile(img, 1)
    # ma = np.percentile(img, 100)
    # normalized = (img - mi) / (ma - mi)
    #
    # normalized = normalized * 255
    # normalized[normalized < 0] = 0
    # end = time.time()
    # global timerc
    # timerc += (end-start)
    # global totalc
    # totalc += 1
    # print(timerc/totalc)
    # normalized = normalized.astype(np.uint8)
    # img = normalized
    # # img = np.stack((img,) * 3, axis=-1)
    # aerial_image.image = img

    # plot_px_distribution(imgpre, img, "POST NORM DISTRIBUTION", 10000)
    bbs = []
    from random import randint
    for hs in aerial_image.hotspots:
        if "new" in hs.status:
            continue
        offset_x = 12
        offset_y = 12
        # offset_x = randint(5, 9)
        # offset_y= randint(5, 9)

        bb = BoundingBox(hs.thermal_loc[0]-offset_x, hs.thermal_loc[1]-offset_y, hs.thermal_loc[0]+offset_x, hs.thermal_loc[1]+offset_y, 1, hs.id)
        if bb.is_partly_within_image(img):
            new = bb.cut_out_of_image(img)
            new.hsId = bb.hsId
            bb = new

        bbs.append(bb)

    tr = TrainingChip.TrainingChip(aerial_image, img.shape, cfg, bbs, (0, 0, 0, 0))
    tr.image = img
    return tr