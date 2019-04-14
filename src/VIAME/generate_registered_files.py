import os

from src.arcticapi import ArcticApi
from src.arcticapi.config import load_config
import pandas as pd
import imgaug as ia
import numpy as np
import glob
from src.arcticapi.csv_parser import parse_ts
out_dir = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/image_registration/out/*.PNG"
cfg = load_config("new_data")

out_ims = glob.glob(out_dir)


api = ArcticApi(cfg)
ims = []
for k in api.rgb_images:
    if api.rgb_images[k].hotspots[0].timestamp is None:
        continue
    ims.append(api.rgb_images[k].hotspots[0])
ims = [x for x in ims if parse_ts(x.timestamp) is not None]
ims.sort(key=lambda x: parse_ts(x.timestamp), reverse=True)
for k in ims:
    ir_path = k.ir.path
    rgb_path = k.rgb.path
    already_complete = False
    for im in out_ims:
        if os.path.basename(rgb_path).split("_")[5] in os.path.basename(im):
            already_complete = True
    if not already_complete:
        with open("/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/input_thermal_images.txt", "a") as myfile:
            myfile.write(ir_path + "\n")
        with open("/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/input_optical_images.txt", "a") as myfile:
            myfile.write(rgb_path + "\n")
    else:
        print("XX")
