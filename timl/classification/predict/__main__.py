import os
import sys
import argparse

import numpy as np
import pandas

from timl.common.datageneration import SkincareDataGenerator
from timl.classification.classifier import Classifier


args_parser = argparse.ArgumentParser(
    description='Automated training and testing of CNNs for multi-class prediction.')

args_parser.add_argument('--keras-model', metavar='<keras_model.h5>', type=str, required=True,
                         help="The keras model to use for testing.")
args_parser.add_argument('--config-table', metavar='<config_table.csv>', type=str, required=True,
                         help="The CSV table with the input used for training  (batch_size, imgfilter, colorfiler, classcolumns,...)")
args_parser.add_argument('--test-dataframe', metavar='<test_dataframe.csv>', type=str, required=True,
                         help="The CSV table with the list of images to test.")
args_parser.add_argument('--img-dir', dest='img_dir', type=str, required=True,
                         help='The directory path in which to look for the images.')
args_parser.add_argument('--output-csv', metavar='<output.csv>', type=str, required=True,
                         help="The CSV table with the list of images to test.")
args_parser.add_argument('--generate-numpy', action="store_true",
                         help='In addition to the output.csv, also binary numpy arrays will be saved.'
                              ' 1) One big numpy file output.npy, 2) a directory output/ containing one .npy array per sample.')
args_parser.add_argument('--cuda-gpu', dest='cuda_gpu', type=int,
                         help='The CUDA GPU number to use for computing predictions')

# TODO -- support also metadata and activations as input
# args_parser.add_argument('--input-type', dest='input_type', type=str, default="images",
#                          help='The input for this model, among: "images", "images_and_metadata", "activations_and_metadata"')
# args_parser.add_argument('--metadata-csv', dest='metadata_csv', type=str,
#                          help='The CSV file containing the metadata for the images')
# args_parser.add_argument('--feature-size', dest='feature_size', type=int,
#                          help='The site of the input features vector')


# This possibly stops the execution if the arguments are not correct
args = args_parser.parse_args()

if args.cuda_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_gpu)

#
# Check for the dataset presence
DATASET_IMG_DIR = args.img_dir
print("Looking for images in directory '{}'".format(DATASET_IMG_DIR))
if not os.path.exists(DATASET_IMG_DIR):
    print("Image dataset dir '{}' doesn't exist".format(DATASET_IMG_DIR))
    sys.exit(10)

if not os.path.isdir(DATASET_IMG_DIR):
    print("Image dataset path '{}' is not a directory".format(DATASET_IMG_DIR))
    sys.exit(10)

model_filename = args.keras_model  # model full filename with path
inference_set_filename = args.test_dataframe  # Name of the dataset
config_table = args.config_table  # The filename for the table with the train output

config_df = pandas.read_csv(config_table)

# Take values from the first line
batch_size = config_df['batchsize'][0]
print("Batch size={}".format(batch_size))
resize_filter = config_df['resizefilter'][0]
class_columns = config_df['classcolumns'][0]
color_space = config_df['colorspace'][0]

#
# load the model
print("Loading the model")
classifier = Classifier()
model = classifier.load_model(model_filename)

#
# Find the image size that the model takes as input
print("Getting image size from layer 0...")
l0 = model.get_layer(index=0)  # type: keras.engine.input_layer.InputLayer
# Get the dimension (the first number is the sample number)
_, w, h, depth = l0.input_shape
img_size = (w, h)
print("Using image size {}".format(img_size))

#
# Create the inference dataframe
if not os.path.exists(inference_set_filename):
    raise Exception("Dataset file '{}' not found.".format(inference_set_filename))

input_df = pandas.read_csv(filepath_or_buffer=inference_set_filename)

# Extract the column names from the config table
if class_columns == "default":
    classes = [c for c in input_df.columns[1:]]
else:
    classes = class_columns.split(';')
n_classes = len(classes)

# n_classes = classifier.get_output_size()
assert n_classes == classifier.get_output_size()

n_samples = len(input_df)
print("# classes: {}, #samples: {}".format(n_classes, n_samples))
print("Classes: ", classes)

#
# Data Generator

# The inference generator must not augment images and must not shuffle
inference_generator = SkincareDataGenerator(
    images_df=input_df,
    images_dir=DATASET_IMG_DIR,
    image_size=img_size,
    resize_filter=resize_filter,
    color_space=color_space,
    batch_size=batch_size,
    image_augmentation="none",
    shuffle=False
)

#
# Run inference
predictions_array = classifier.predict(generator=inference_generator)

assert predictions_array.shape[0] == n_samples
assert predictions_array.shape[1] == n_classes

#
# Compose output dataframes
predictions_df = pandas.DataFrame(data=predictions_array, columns=classes)
image_names_df = input_df[['image_name']]
output_df = pandas.concat([image_names_df, predictions_df], ignore_index=False, axis=1)  # type: pandas.DataFrame

#
# SAVE
#

# As CSV
print("Saving CSV...")
output_df.to_csv(args.output_csv, header=True, index=False)

if args.generate_numpy:

    # As single numpy
    print("Saving single numpy...")
    output_predictions_filename = os.path.splitext(args.output_csv)[0] + ".npy"
    print("Writing predictions to {}".format(output_predictions_filename))
    np.save(output_predictions_filename, predictions_array)

    # As separate numpies in a directory
    print("Saving multiple numpies...")
    output_predictions_dir = os.path.splitext(args.output_csv)[0]
    if not os.path.exists(output_predictions_dir):
        os.mkdir(output_predictions_dir)
    for i, img_name in enumerate(input_df['image_name']):
        # Save using the format image_name-augmentation.npy, where augmentation is 10 digits.
        numpy_file_name = "{}-{:010d}.npy".format(img_name, 0)
        np.save(os.path.join(output_predictions_dir, numpy_file_name), predictions_array[i])

print("All done.")
