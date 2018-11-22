import argparse
import sys
from arcticapi import ArcticApi
from arcticapi.config import load_config, make_config


# Preprocess is an api with many features to crop, augment, and create darknet formatted labels for training.
# There are many argument options you can use to augment the data.  arcticapi.crop is where all of the actual cropping
# happens so if you are interested in the implementation check it out.  When running this for the first time I would reccomend
# using the -d debug flag as it paints the bounding box so you can see if the augmentation values you've chosen don't break the system.
# It's pretty good at figuring out how to crop but there are probably some issues that will occur if you start to use crop/shift values
# that are close to the total height/width of the images.
cfg = None  # CropCfg to be

def main():
    parser = argparse.ArgumentParser(description='Command line interface for cropping seal data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cfg', action="store")
    args = parser.parse_args(sys.argv[1:1])

    if args.cfg is not None:
        cfg = load_config(args.cfg)
        if cfg is None:
            print("Config " + args.conf + " was not found.")


    else:
        del parser
        parser = argparse.ArgumentParser(description='Command line interface for cropping seal data.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--csv', required=True, type=str, default=None,
                            help='csv file: relative path to the seal image data csv file')
        parser.add_argument('--imdir', required=True, type=str, default=None,
                            help='image dir: relative path to the directory containing all images')
        parser.add_argument('--imout', required=True, type=str, default=None,
                            help='out dir: relative path to the directory to store cropped images')
        parser.add_argument('--name', type=str, default="last_run", help='name: name for cfg')
        parser.add_argument('--bb', type=int, default=70,
                            help='bounding box size: size of bounding box width and height around the center point')
        parser.add_argument('--min', type=int, default=100,
                            help='min shift: min value shift center point dx and dy, calculated as random value between min and max')
        parser.add_argument('--max', type=int, default=250,
                            help='max shift: max value shift center point dx and dy, calculated as random value between min and max')
        parser.add_argument('--cs', type=int, default=512, help='crop size: size of croped region')
        parser.add_argument('-c', action='store_true', default=False,
                            help='global seal class: puts all seals as one class')
        parser.add_argument('-b', action='store_true', default=False, help='make bear crops/labels')
        parser.add_argument('-a', action='store_true', default=False, help='make anomaly crops/labels')
        parser.add_argument('-d', action='store_true', default=False,
                            help='debug: draws bounding box bounds NOT FOR TRAINING')

        # Parse
        args = parser.parse_args(sys.argv[1:])

        # Make CropCfg
        make_config(args)
        cfg = load_config(args.name)

    if cfg == None:
        print("Error generating or getting the configuration")
        return

    if args.cs % 32 != 0:
        print(
            "WARNING - the crops size is not divisible by 32, for best results on small objects with YOLO image width or height should be divisible by 32.")

    api = ArcticApi(cfg.csv, cfg.im_dir)
    api.crop_label_all(cfg)



if __name__== "__main__":
  main()