# Tools to split a 1-hot dataset (as ISIC 2019 challenge) into train/dev/test
# It is not pure random sampling: it preserves the distribution of classes.

import sys
import os
import argparse

import pandas

args_parser = argparse.ArgumentParser(
    description='Tool to split a 1-hot dataset (CSV format, as ISIC 2019 challenge) into train/dev/test.'
                ' It is not pure random sampling: it preserves the distribution of classes.')

args_parser.add_argument('input_csv', metavar='<input_dataframe.csv>', type=str,
                         help="The dataset to split. The first column is assumed to be the image name."
                              "The remaining columns are 1-hot encoded classes.")
args_parser.add_argument('dev_prop', metavar='<dev_proportion:float>', type=float,
                         help="The proportion of samples for the dev set. Usually between 0.0 and around 0.1.")
args_parser.add_argument('test_prop', metavar='<test_proportion:float>', type=float,
                         help="The proportion of samples for the test set. Usually between 0.0 and around 0.1.")

args = args_parser.parse_args()


dataset_filepath = args.input_csv
dev_prop = args.dev_prop
test_prop = args.test_prop

#
# get DATAFRAME
print("Loading ''".format(dataset_filepath))
if not os.path.exists(dataset_filepath):
    print("File '{}' doesn't exists.".format(dataset_filepath))
    exit(10)

if not dataset_filepath.endswith(".csv"):
    print("Please, specify a .csv file")
    exit(10)

dataset = pandas.read_csv(dataset_filepath)
n_samples = len(dataset)
print("#samples={}".format(n_samples))

#
# get PROPORTIONS
print("Trying to extract proportions of {} and {} lines from dataset".format(dev_prop, test_prop))

if dev_prop + test_prop > 1.0:
    print("Sum of extraction proportions can not be more than 1. Found {} + {}".format(dev_prop, test_prop))
    exit(10)


#
# dataset INFO
n_columns = dataset.shape[1]
n_classes = n_columns - 1

# Convert to simple str list
classes = [str(c) for c in dataset.columns[1:]]

assert len(classes) == n_classes

print("#columns={}".format(n_columns))
print("#classes={}".format(n_classes))
print("Classes:")
print(classes)


#
# Count the occurrencies of each class
# This dataset/dictionary counts the occurrencies for each class
freq_count = dataset[classes].sum(axis=0)
print("Frequency count:")
print(freq_count)

#
# Build the extraction
train_df = pandas.DataFrame(columns=dataset.columns)
dev_df = pandas.DataFrame(columns=dataset.columns)
test_df = pandas.DataFrame(columns=dataset.columns)


# For each class
for cls in classes:
    print("For class {}".format(cls))

    #
    # calc how many samples we want
    samples_per_class = freq_count[cls]

    #
    # extract df from global

    # It used to be "select": https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select.html#pandas.DataFrame.select
    # On the column of the wanted class
    # Create the boolean mask with True if the value is 1
    cls_mask = dataset[cls].map(lambda x: x == 1)
    # Use the true/false series to filter the needed indices
    cls_df = dataset.loc[cls_mask]  # type: pandas.DataFrame

    print("Extracted {} samples (Expected {})".format(len(cls_df), samples_per_class))

    assert len(cls_df) == samples_per_class

    n_dev_samples_to_extract = int(samples_per_class * dev_prop)
    n_test_samples_to_extract = int(samples_per_class * test_prop)

    print("Extracting {} dev samples".format(n_dev_samples_to_extract))
    print("Extracting {} test samples".format(n_test_samples_to_extract))

    # sample dev
    dev_samples = cls_df.sample(n_dev_samples_to_extract)
    cls_df = cls_df.loc[~ cls_df.index.isin(dev_samples.index)]

    # sample test
    test_samples = cls_df.sample(n_test_samples_to_extract)
    cls_df = cls_df.loc[~ cls_df.index.isin(test_samples.index)]

    print("Remaining {} train samples".format(len(cls_df)))

    # concat with output dfs
    train_df = pandas.concat([train_df, cls_df])
    dev_df = pandas.concat([dev_df, dev_samples])
    test_df = pandas.concat([test_df, test_samples])

#
# Resulting DFs stats
print("================")
print("Train DF size: {}".format(train_df.shape))
print("Dev DF size: {}".format(dev_df.shape))
print("Test DF size: {}".format(test_df.shape))

#
# Compose output filename
root_filename, _ = os.path.splitext(dataset_filepath)


save_data = (
    (train_df, root_filename + "-train.csv"),
    (dev_df, root_filename + "-dev.csv"),
    (test_df, root_filename + "-test.csv")
)


for (df, filename) in save_data:
    print("Saving to {}".format(filename))

    with open(filename, "w") as f:
        df.to_csv(f, index=False, header=True)


print("All done.")
