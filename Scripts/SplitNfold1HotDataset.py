# Tools to split a 1-hot dataset (as ISIC 2019 challenge) into train/dev/test
# It is not pure random sampling: it preserves the distribution of classes.

import sys
import os
import argparse

import pandas

N_FOLDS_DEFAULT = 5

args_parser = argparse.ArgumentParser(
    description='Tool to split a 1-hot dataset (CSV format, as ISIC 2019 challenge) into train/dev/test.'
                ' It is not pure random sampling: it preserves the distribution of classes.'
                'Output files name will have the following template <in_dataset>-foldNN-[train|dev|test].csv')

args_parser.add_argument('input_csv', metavar='<input_dataframe.csv>', type=str,
                         help="The dataset to split. The first column is assumed to be the image name."
                              "The remaining columns are 1-hot encoded classes.")
args_parser.add_argument('--dev-prop', type=float, default=None,
                         help="The proportion of samples for the dev set. Usually around 0.1."
                              " If not specified the same number of samples used for testing will be used for dev.")
args_parser.add_argument('--folds', type=int, default=N_FOLDS_DEFAULT,
                         help="The number of folds (default {})".format(N_FOLDS_DEFAULT))
args_parser.add_argument('--out-dir', type=str, default=None,
                         help="Specifies the output directory of the folds")

args = args_parser.parse_args()


dataset_filepath = args.input_csv
dev_prop = args.dev_prop
n_folds = args.folds
out_dir = args.out_dir

if n_folds < 2:
    print("Number of folds must be at least 2. Found {}.".format(n_folds))
    exit(10)

print("Splitting in {} folds...".format(n_folds))

if out_dir is not None:
    if not os.path.exists(out_dir):
        raise Exception("Output directory '{}' doesn't exist.".format(out_dir))


if dev_prop is not None:
    if not 0 <= dev_prop < 1.0:
        raise Exception("dev-prop must lie in range [0,1). Found {}.".format(dev_prop))

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
# Build the final datasets
# For each fold, prepare train/dev/and test datasets
# Each element will contain a 3-tuple of datasets (train,dev,test)
fold_dfs = []

for f in range(n_folds):
    train_df = pandas.DataFrame(columns=dataset.columns)
    dev_df = pandas.DataFrame(columns=dataset.columns)
    test_df = pandas.DataFrame(columns=dataset.columns)

    fold_dfs.append((train_df, dev_df, test_df))

#
# For each class
for cls in classes:
    print("For class {}".format(cls))

    # Create the boolean mask with True if the value is 1
    cls_mask = dataset[cls].map(lambda x: x == 1)
    # Use the true/false series to filter the needed indices
    cls_df = dataset.loc[cls_mask]  # type: pandas.DataFrame
    # shuffle the elements, because we are going to take the test samples sequentially.
    cls_df = cls_df.sample(frac=1, random_state=42)
    # Re-compute the index as 0..n_samples, so we can more easily address the samples for each folder
    cls_df = cls_df.reset_index(drop=True)
    n_class_samples = freq_count[cls]
    assert len(cls_df) == n_class_samples
    print("Extracted {} samples".format(len(cls_df)))

    #
    # Each fold will identify the samples used to test
    # in the range [first_index,last_index)
    for f in range(n_folds):
        first_index = int(f * n_class_samples / n_folds)
        last_index = int((f+1) * n_class_samples / n_folds)

        #
        # Take the test samples
        n_test_samples = last_index - first_index
        print("Extracting {} test samples in indices {} to {}".format(n_test_samples, first_index, last_index))
        test_samples = cls_df.loc[first_index:last_index-1]
        assert n_test_samples == len(test_samples)

        other_samples = cls_df.loc[~ cls_df.index.isin(test_samples.index)]
        n_other_samples = n_class_samples - n_test_samples
        assert len(other_samples) == n_other_samples

        #
        # Sample dev
        if dev_prop is None:
            n_dev_samples = n_test_samples
        else:
            n_dev_samples = int(n_other_samples * dev_prop)

        dev_samples = other_samples.sample(n_dev_samples)
        assert n_dev_samples == len(dev_samples)

        #
        # Take the leftovers as training set
        train_samples = other_samples.loc[~ other_samples.index.isin(dev_samples.index)]
        n_train_sample = n_other_samples - n_dev_samples
        assert n_train_sample == len(train_samples)

        print("Extracted {}/{}/{} train/dev/test samples".format(n_train_sample, n_dev_samples, n_test_samples))

        #
        # Concat with output dfs

        # Retrieve the sets according to the fold number
        train_df, dev_df, test_df = fold_dfs[f]

        train_df = pandas.concat([train_df, train_samples])
        dev_df = pandas.concat([dev_df, dev_samples])
        test_df = pandas.concat([test_df, test_samples])

        fold_dfs[f] = train_df, dev_df, test_df


#
# Compose output filename
root_filename, _ = os.path.splitext(dataset_filepath)

if out_dir is not None:
    # substitute the target directory
    _, last_path = os.path.split(root_filename)
    root_filename = os.path.join(out_dir, last_path)

print("Saving with root '{}'...".format(root_filename))

for f in range(n_folds):
    fold_filename = root_filename + "-fold{:02d}".format(f)

    train_df, dev_df, test_df = fold_dfs[f]

    print("Saving fold {} with {}/{}/{} samples...".format(fold_filename,
                                                        len(train_df),
                                                        len(dev_df),
                                                        len(test_df)))

    save_data = (
        (train_df, fold_filename + "-train.csv"),
        (dev_df, fold_filename + "-dev.csv"),
        (test_df, fold_filename + "-test.csv")
    )

    for (df, filename) in save_data:
        with open(filename, "w") as f:
            df.to_csv(f, index=False, header=True)


print("All done.")
