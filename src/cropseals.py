import argparse
from arcticapi import ArcticApi

import argparse
parser = argparse.ArgumentParser(description='SEALNET.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='interger list')
parser.add_argument('--sum', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.sum(args.integers))
args = parser.parse_args()
print(args)
print(args.accumulate(args.integers))