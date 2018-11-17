import argparse
import sys
from arcticapi import ArcticApi

parser = argparse.ArgumentParser(description='Command line interface for cropping seal data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--csv', required=True, type=str, default=None, help='csv file: relative path to the seal image data csv file')
parser.add_argument('--imdir', required=True, type=str, default=None, help='image dir: relative path to the directory containing all images')
parser.add_argument('--out', required=True, type=str, default=None, help='out dir: relative path to the directory to store cropped images')
parser.add_argument('--bb', type=int, default=70, help='bounding box size: size of bounding box width and height around the center point')
parser.add_argument('--min', type=int, default=100, help='min shift: min value shift center point dx and dy, calculated as random value between min and max')
parser.add_argument('--max', type=int, default=250, help='max shift: max value shift center point dx and dy, calculated as random value between min and max')
parser.add_argument('--label', type=str, default="training_list.txt", help='label: output file with all absolute label paths for training')
# Parse
args = parser.parse_args(sys.argv[1:])

api = ArcticApi(args.csv, args.imdir)
api.crop_label_all(args.out, args.bb, args.min, args.max, args.label)

print(args)