from arcticapi import ArcticApi
from arcticapi.augmnetation import AugRgb
from arcticapi.config import CropCfg
from arcticapi.visuals import print_loading_bar
import os

csv = '/Users/yuval/Documents/XNOR/Bears-n-Seals/src/bbox-labeler/out.csv'
img_path = '/Users/yuval/Documents/XNOR/Bears-n-Seals/images/CHESS/'
out_path = "bbox-labeler/relabel/"
if not os.path.exists(out_path):
    os.makedirs(out_path)

api = ArcticApi(csv, img_path)
cfg =  CropCfg(csv, img_path, out_path, 100, 0, 0, 640, "re-label.txt", False, False, False, True, "rgb", "relabel", False, False)
chips = []
for image_path in api.images:
    aeral_image = api.images[image_path]
    bounding_boxes = aeral_image.getBboxesForReLabeling(cfg)
    chips = chips + AugRgb.prepare_chips(cfg, aeral_image, bounding_boxes)

non_updated_chips = []
for chip in chips:
    for bbox in chip.bboxes.bounding_boxes:
        hs = api.hsm.get_hs(bbox.hsId)
        assert(hs is not None) # IMPORTANT hs never should be none
        if not hs.updated:
            non_updated_chips.append(chip)
            break

print("\nOriginal Stats:")
AugRgb.print_bbox_stats(non_updated_chips)

for idx, chip in enumerate(non_updated_chips):
    print_loading_bar(((idx + 0.0) / len(non_updated_chips)) * 100.0)
    if not chip.load():
        print("Chip not loaded in api.py :(")
        continue
    chip.load()  # load image
    chip.save()  # save image and labels
    chip.free()  # free image

    print("COMPLETE")
