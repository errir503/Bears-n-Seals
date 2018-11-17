import sys

from arcticapi import ArcticApi

config = {
    "output_dir": "images/results/",
    "offset": 80
}

def main():
    csv_file = sys.argv[1]
    api = ArcticApi(csv_file)
    for hotspot in api.hsm.hotspots:
        if hotspot.classIndex != 4:
            hotspot.genCropsAndLables(100, 250)
    # visuals.show_ir(hsm)
    api.register()
    # api.prep_label("images/results/", 80, 16908)
    # crop_hotspots(hsm)
    # align_images(hotspots)


# Call main function
if __name__ == '__main__':
    main()
