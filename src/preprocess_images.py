import sys

from arcticapi import ArcticApi
from arcticapi.crop import CropCfg


def main():
    csv_file = sys.argv[1]
    imdir = sys.argv[2]
    out_path = sys.argv[3]
    api = ArcticApi(csv_file, imdir)
    api.crop_label_all(CropCfg(out_path, 70, 100, 250, 800, "a.txt", True, False, False))

    # visuals.show_ir(hsm)
    # api.register()
    # api.prep_label("images/results/", 80, 16908)
    # crop_hotspots(hsm)
    # align_images(hotspots)


# Call main function
if __name__ == '__main__':
    main()
