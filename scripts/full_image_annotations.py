import cv2
import os
from src.arcticapi import ArcticApi
from src.arcticapi.config import load_config
from random import randint
import numpy as np

from src.arcticapi.model.HotSpot import ColorsList

out_path = "/data/raw_data/merged.csv"
cfg = load_config("new_data")

api = ArcticApi(cfg)
i = 0
ir_paths = []
for path in api.rgb_images:
    if i > 40:
        break
    base = os.path.basename(path)
    base = str(i) + "_"
    aerial_image = api.rgb_images[path]
    good_im = True
    for hs in aerial_image.hotspots:
        if not hs.updated:
            good_im = False
        if not hs.classIndex < 3:
            good_im = False
        if "new" in hs.status:
            good = False
    if not good_im:
        continue
    if len(aerial_image.hotspots) < 3:
        continue
    if not aerial_image.load_image():
        continue
    if not aerial_image.hotspots[0].ir.load_image():
        continue
    im = aerial_image.image
    i += 1
    ir_im = aerial_image.hotspots[0].ir.image
    ir_im = np.stack((ir_im,) * 3, axis=-1)
    mi = np.percentile(ir_im, 1)
    ma = np.percentile(ir_im, 100)
    normalized = (ir_im - mi) / (ma - mi)

    normalized = normalized * 255
    normalized[normalized < 0] = 0
    normalized = normalized.astype(np.uint8)


    for hs in aerial_image.hotspots:
        color = ColorsList[hs.classIndex]
        hs.updated_left = hs.updated_left - randint(1, 5)
        hs.updated_right = hs.updated_right + randint(1, 5)
        hs.updated_bot = hs.updated_bot + randint(1, 5)
        hs.updated_top = hs.updated_top - randint(1, 5)
        cv2.rectangle(im, (hs.updated_left, hs.updated_top),
                      (hs.updated_right, hs.updated_bot),
                      color, 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        margin = 3
        thickness = 2
        size = cv2.getTextSize(hs.species, font, font_scale, thickness)

        # cv2.rectangle(im, (hs.updated_left, hs.updated_top+3), (hs.updated_left+size[0][0]+2*margin, hs.updated_top-size[0][1]-margin*2), color, -1)
        # cv2.putText(im, hs.species, (hs.updated_left+margin, hs.updated_top-3), font, font_scale, (255, 255, 255), thickness = thickness)

        cv2.rectangle(normalized, (hs.thermal_loc[0] - randint(5,8), hs.thermal_loc[1] - randint(5,8)),
                      (hs.thermal_loc[0] + randint(5,8), hs.thermal_loc[1] + randint(5,8)),
                      color, 1)
    cv2.imwrite('/fast/res/'+base + "color.jpg", im)
    cv2.imwrite('/fast/res/'+base + "thermal.jpg", normalized)

    aerial_image.free()
    aerial_image.image = None
    im = None

pass
