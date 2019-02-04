import os

from arcticapi import ArcticApi
from src.arcticapi.config import load_config

csv = '/home/yuval/Documents/XNOR/Bears-n-Seals/src/bbox-labeler/updated_live_out.csv'
img_path = '/data/CHESS/'
out_path = "bbox-labeler/relabel/"
cfg = load_config("full")

if not os.path.exists(out_path):
    os.makedirs(out_path)

api = ArcticApi(csv, img_path)

for img_name in api.rgb_images:
    aerial_image = api.rgb_images[img_name]
    aerial_image.saveLabels(cfg)
    aerial_image.free()