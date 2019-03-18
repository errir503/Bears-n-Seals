from src.arcticapi import ArcticApi
from src.arcticapi.config import load_config


cfg = load_config("rgb")
# cfg.csv = "data/_CHESS_ImagesSelected4Detection.csv"
api = ArcticApi(cfg)
rgb_list_path = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/input_optical_images.txt"
ir_list_path = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/input_thermal_images.txt"
for rgb_path in api.rgb_images:
    aerial_im = api.rgb_images[rgb_path]
    good_im = True
    for hs in aerial_im.hotspots:
        if hs.classIndex > 2:
            good_im = False
        if not hs.updated:
            good_im = False
        if "bad_res" in hs.status:
            good_im = False
    if not good_im:
        continue

    ir_path = aerial_im.hotspots[0].ir.path
    with open(rgb_list_path, 'a') as the_file:
        the_file.write(rgb_path + '\n')
    with open(ir_list_path, 'a') as the_file:
        the_file.write(ir_path + '\n')