import os

from arcticapi import ArcticApi

csv = '/Users/yuval/Documents/XNOR/Bears-n-Seals/src/bbox-labeler/out.csv'
img_path = '/Users/yuval/Documents/XNOR/Bears-n-Seals/images/CHESS/'
out_path = "bbox-labeler/relabel/"
if not os.path.exists(out_path):
    os.makedirs(out_path)

api = ArcticApi(csv, img_path)

for img_name in api.rgb_images:
    aerial_image = api.rgb_images[img_name]
    for hs in aerial_image.hotspots:
        pass