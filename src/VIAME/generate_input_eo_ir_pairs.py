import os

from src.arcticapi import ArcticApi
from src.arcticapi.config import load_config
import pandas as pd
import imgaug as ia
import numpy as np
import glob
from src.arcticapi.csv_parser import parse_ts
cfg = load_config("new_data")


api = ArcticApi(cfg)
ims = []
for k in api.rgb_images:
    has_seal = False
    skip = False
    for hs in api.rgb_images[k].hotspots:
        if hs.classIndex < 2:
            has_seal = True
        if "bad_res" in hs.status:
            skip = True

    if has_seal and not skip:
        ims.append(k)

for k in ims:

    rgb_path = api.rgb_images[k].path
    ir_path = api.rgb_images[k].hotspots[0].ir.path
    already_complete = False
    with open("/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/input_thermal_images.txt", "a") as myfile:
        myfile.write(ir_path + "\n")
    with open("/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/input_optical_images.txt", "a") as myfile:
        myfile.write(rgb_path + "\n")

