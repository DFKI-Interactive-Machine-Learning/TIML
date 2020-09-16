import argparse
import os
import sys

from timl.common.utils import generate_random_image

args_parser = argparse.ArgumentParser(
    description='Generation of random noise images.')

args_parser.add_argument('output_dir', type=str,
                         help="The destination directory.")
args_parser.add_argument('count', type=int,
                         help="The number of images to generate.")
args_parser.add_argument('width', type=int,
                         help='The width of the generate images.')
args_parser.add_argument('height', type=int,
                         help='The height of the generate images.')

# This possibly stops the execution if the arguments are not correct
args = args_parser.parse_args()

if not os.path.exists(args.output_dir):
    print("Destination dir '{}' doesn't exist".format(args.output_dir))
    sys.exit(10)

if not os.path.isdir(args.output_dir):
    print("Destination path '{}' is not a directory".format(args.output_dir))
    sys.exit(10)

n = args.count

print("Generating {} images of resolution {}x{} into {}.".format(n, args.width, args.height, args.output_dir))

for i in range(n):
    print(i)
    img = generate_random_image((args.width, args.height))
    img.save(os.path.join(args.output_dir, "rndimage-{:07d}.jpg".format(i)))
