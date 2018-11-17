import sys

from arcticapi import ArcticApi

def main():
    csv_file = sys.argv[1]
    res_path = sys.argv[2]
    out_path = sys.argv[2]
    api = ArcticApi(csv_file, res_path)
    api.crop_label_all(out_path, width_bb=70, minShift=100, maxShift=250, label=True)

    # visuals.show_ir(hsm)
    api.register()
    # api.prep_label("images/results/", 80, 16908)
    # crop_hotspots(hsm)
    # align_images(hotspots)


# Call main function
if __name__ == '__main__':
    main()
