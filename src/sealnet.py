import argparse


parser = argparse.ArgumentParser(description='SEALNET')

parser.add_argument('--crop', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args)
print(args.accumulate(args.integers))