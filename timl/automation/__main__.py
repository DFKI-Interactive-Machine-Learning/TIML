import sys
import os
import gc
import datetime
import platform
import re
import argparse
import shutil

import pandas

import tensorflow as tf

from timl.classification.classifier import TrainResults
from timl.classification.classifier_factory import make_classifier
from ..common.datageneration import SkincareDataGenerator
from ..common.plotting import generate_trainval_plot, generate_accval_plot, generate_ROC_plot
from ..common.utils import format_array
from timl.classification.inspect.common import inspect_predictions

from timl.networking.config import get_isic_base_path

from typing import List
from typing import Dict
from typing import Optional

DF_INPUT_COLUMNS = ["method", "dataset", "split", "epochs", "imgaug", "batchsize",
                    "imgsize", "resizefilter", "colorspace",
                    "classcolumns", "classweights"]

# The column names that must be present in the provided input dataframe
DF_OUTPUT_COLUMNS = ["trainsize", "augmult", "malprop", "trainstart", "traintime", "trainacc", "trainmodel",
                     "valsize", "valmalprop", "valacc", "valspec", "valsens",
                     "testsize", "testmalprop", "testacc", "testspec", "testsens", "testauc",
                     "exceptions", "notes"]

DF_COLUMNS = DF_INPUT_COLUMNS + DF_OUTPUT_COLUMNS


#
# Config stuff
DATASET_IMG_DIR = get_isic_base_path()


#
# Types for the train input configuration
# Useful to force the type in case of empty cells or for names containing only digits.
DATA_DFs_DTYPES = {
    "image_name": str,
    "classcolumns": str
}

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(
        description='Automated training and testing of CNNs for multi-class prediction.')

    args_parser.add_argument('input_table', metavar='<input_table.csv>', type=str,
                             help="The CSV table with the input information.")
    args_parser.add_argument('--img-dir', dest='img_dir', type=str,
                             help='The directory path in which to look for the images.'
                                  ' If omitted, uses the one specified in skincare_config.json')
    args_parser.add_argument('--out-dir', dest='out_dir', type=str,
                             help='The directory path in which the models will be written.'
                                  ' If omitted, creates one using the "skincare_train_output_yyyymmdd-hhmmss" template')
    args_parser.add_argument('--cuda-gpu', dest='cuda_gpu', type=int,
                             help='The CUDA GPU number to use for training')
    args_parser.add_argument('--skip-roc', action='store_true',
                             help="Skip plotting the ROC graphs (and the computation of AUCs).")

    # This possibly stops the execution if the arguments are not correct
    args = args_parser.parse_args()

    if args.cuda_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_gpu)

    # Overrides the default images dir
    if args.img_dir is not None:
        DATASET_IMG_DIR = args.img_dir

    #
    # Check for the dataset presence
    print("Looking for images in directory '{}'".format(DATASET_IMG_DIR))
    if not os.path.exists(DATASET_IMG_DIR):
        print("Image dataset dir '{}' doesn't exist".format(DATASET_IMG_DIR))
        sys.exit(10)

    if not os.path.isdir(DATASET_IMG_DIR):
        print("Image dataset path '{}' is not a directory".format(DATASET_IMG_DIR))
        sys.exit(10)

    input_csv_filename = args.input_table
    if not os.path.exists(input_csv_filename):
        raise Exception("File {} doesn't exist.".format(input_csv_filename))

    input_df = pandas.read_csv(input_csv_filename)  # type: pandas.DataFrame

    #
    # The input dataframe MUST contain ALL of the columns registered here.
    input_column_names = list(input_df.columns)  # type: List[str]
    for col_name in DF_COLUMNS:
        if col_name not in input_column_names:
            raise Exception("Column {} not found in dataframe".format(col_name))

    #
    # Tech stuff. Count CPUs
    # Multiprocessing parameters
    # See: https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
    # See: https://keras.io/models/sequential/
    n_cpus = os.cpu_count()  # might be None

    if n_cpus is None:
        print("WARNING! Can not read the number of CPUs. Setting to 1. Data Generation will be single-threaded.")
        n_cpus = 1
    else:
        print("Found {} CPUs".format(n_cpus))

    #
    # Create the output directory

    if args.out_dir is None:
        # This is the timestamp
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d-%H%M%S")
        out_dirname = "skincare_train_output-{}".format(timestamp_str)
        print("Creating dir '{}'".format(out_dirname))
        os.mkdir(out_dirname)
    else:
        out_dirname = args.out_dir
        if not os.path.exists(out_dirname):
            os.mkdir(out_dirname)

    # Copy input file
    shutil.copyfile(src=input_csv_filename, dst=os.path.join(out_dirname, "automation_input.csv"))

    #
    # Initialize the output
    # Each element of this list is a dictionary.
    # Each dictionary will contain the data for an output line.
    out_dicts = []  # type: List[Dict]

    # For each row, do the tests
    for idx in input_df.index:

        print("TRAIN #{}".format(idx))

        row = input_df.iloc[idx]
        # print("{} - {}".format(idx, row))

        #
        # Read all input data
        train_method = row["method"]

        input_dataset_rootname = row["dataset"]
        split_policy = row["split"]

        train_epochs = int(row["epochs"])
        image_augmentation = row["imgaug"]
        batch_size = int(row["batchsize"])
        image_size = int(row["imgsize"])
        resize_filter = (row["resizefilter"])
        color_space = row["colorspace"]
        class_columns = row["classcolumns"]
        class_weights = row["classweights"]

        #
        # Setup the row of the output dataframe
        out_dict = {
            "method": train_method,
            "dataset": input_dataset_rootname,
            "split": split_policy,
            "epochs": train_epochs,
            "imgaug": image_augmentation,
            "batchsize": batch_size,
            "imgsize": image_size,
            "resizefilter": resize_filter,
            "colorspace": color_space,
            "classcolumns": class_columns,
            "classweights": class_weights,
        }

        #
        # Get NOTES
        notes = []  # type: List[str]
        # Empty cells are returned as NaN. Forcing conversion to str.
        old_notes = row["notes"] if isinstance(row["notes"], str) else ""

        notes.append(old_notes)

        # Report on tech stuff
        notes.append("Training on {}".format(platform.node()))
        notes.append("#CPUs: {}.".format(n_cpus))

        # Prepare for exception messages
        exceptions = ""

        # Declare it outside of the try block, because we might need to manually delete its model.
        train_res = None  # type: Optional[TrainResults]

        #
        # Explicitly create a session for this training iteration, so we can clean it up at the end of the cycle.
        tf_session = tf.Session(graph=tf.Graph())
        with tf_session.as_default():
            with tf_session.graph.as_default():

                try:
                    # Load train/val/test from files.
                    split_policy = str(split_policy)
                    if split_policy == "pre":

                        #
                        # Compose the name of the files with the data
                        train_set_filename = os.path.join(input_dataset_rootname + "-train.csv")
                        dev_set_filename = os.path.join(input_dataset_rootname + "-dev.csv")
                        test_set_filename = os.path.join(input_dataset_rootname + "-test.csv")

                        for fn in [train_set_filename, dev_set_filename, test_set_filename]:
                            if not os.path.exists(fn):
                                raise Exception("Dataset file '{}' not found.".format(fn))

                        #
                        # Create the dataframes
                        train_df = pandas.read_csv(filepath_or_buffer=train_set_filename, dtype=DATA_DFs_DTYPES)
                        dev_df = pandas.read_csv(filepath_or_buffer=dev_set_filename, dtype=DATA_DFs_DTYPES)
                        test_df = pandas.read_csv(filepath_or_buffer=test_set_filename, dtype=DATA_DFs_DTYPES)

                    elif split_policy == "list":
                        set_filenames = input_dataset_rootname.split(";")

                        if len(set_filenames) < 2:
                            raise Exception("Must specify at least 2 datasets (train/dev). Found {}".format(len(set_filenames)))

                        train_df = pandas.read_csv(filepath_or_buffer=set_filenames[0], dtype=DATA_DFs_DTYPES)
                        dev_df = pandas.read_csv(filepath_or_buffer=set_filenames[1], dtype=DATA_DFs_DTYPES)

                        if len(set_filenames) > 2:
                            test_df = pandas.read_csv(filepath_or_buffer=set_filenames[2], dtype=DATA_DFs_DTYPES)
                        else:
                            test_df = None

                    # On-the-fly split the dataset
                    else:
                        dataset_filename = input_dataset_rootname + ".csv"

                        if not os.path.exists(dataset_filename):
                            raise Exception("Dataset file '{}' not found.".format(dataset_filename))

                        full_dataset = pandas.read_csv(dataset_filename,
                                                       dtype=DATA_DFs_DTYPES)  # type: pandas.DataFrame

                        #
                        # Fraction split
                        if re.match("frac=(.+)", split_policy):
                            res = re.match("frac=(.+)", split_policy)
                            # Must be a float number
                            split_proportion = float(res.group(1))

                            if split_proportion <= 0.0 or split_proportion >= 0.5:
                                raise Exception("Too many elements selected for validation and training. "
                                                "You must set 0 < split < 0.5, found {}.".format(split_proportion))

                            dev_and_test_size = int(len(full_dataset) * split_proportion)

                        #
                        # Split specifying the number of samples
                        elif re.match("n=(.+)", split_policy):
                            res = re.match("n=(.+)", split_policy)

                            dev_and_test_size = int(res.group(1))

                        else:
                            raise Exception("Unrecognized split mode {}.".format(split_policy))

                        print("Extracting 2X {} samples for validation and test.".format(dev_and_test_size))

                        # See: https://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe
                        # E.g.: df_rest = df.loc[~df.index.isin(df_percent.index)]
                        dev_df = full_dataset.sample(n=dev_and_test_size)  # type: pandas.DataFrame
                        full_dataset = full_dataset.loc[~full_dataset.index.isin(dev_df.index), :]
                        test_df = full_dataset.sample(n=dev_and_test_size)  # type: pandas.DataFrame
                        train_df = full_dataset.loc[~full_dataset.index.isin(test_df.index), :]  # type: pandas.DataFrame

                        # We need to have sequential indices in these dataframes
                        train_df.reset_index(drop=True, inplace=True)
                        dev_df.reset_index(drop=True, inplace=True)
                        test_df.reset_index(drop=True, inplace=True)

                    #
                    # Consistency check. The three dataframes must not overlap
                    train_intersect_merge = train_df.merge(dev_df, how='inner', on=['image_name'])
                    if len(train_intersect_merge) > 0:
                        notes.append("Warning train and dev set overlap of {} elements!".format(len(train_intersect_merge)))

                    if test_df is not None:
                        train_intersect_test = train_df.merge(test_df, how='inner', on=['image_name'])
                        if len(train_intersect_test) > 0:
                            notes.append("Warning train and test set overlap of {} elements!".format(len(train_intersect_test)))

                        dev_intersect_test = dev_df.merge(test_df, how='inner', on=['image_name'])
                        if len(dev_intersect_test) > 0:
                            notes.append("Warning dev and test set overlap of {} elements!".format(len(dev_intersect_test)))

                    #
                    # Take the list and number of columns to analyze
                    class_columns = str(class_columns).strip()  # type: Optional[str]
                    if class_columns == "default":
                        columns_list = [c for c in train_df.columns[1:]]
                    elif class_columns == "ben_mal" or class_columns == "" or class_columns == 'nan':
                        columns_list = ["ben_mal"]
                    else:
                        columns_list = class_columns.split(';')

                    n_output_classes = len(columns_list)
                    print("Performing classification on {} classes.".format(n_output_classes))

                    # The classifier
                    classifier = make_classifier(train_method=train_method, image_size=image_size,
                                                 n_classes=n_output_classes)

                    #
                    # For multi-label classification
                    is_multilabel = classifier.is_multilabel()
                    print("Is a multi-label classifier: {}".format(is_multilabel))

                    #
                    # Instantiate data generators
                    train_generator = SkincareDataGenerator(
                        images_df=train_df,
                        images_dir=DATASET_IMG_DIR,
                        image_size=(image_size, image_size),
                        resize_filter=resize_filter,
                        color_space=color_space,
                        batch_size=batch_size,
                        image_augmentation=image_augmentation,
                        truth_df=train_df,
                        truth_columns=columns_list,
                        shuffle=True,
                        is_multilabel=is_multilabel
                    )

                    out_dict.update({
                        "trainsize": len(train_df),
                        "malprop": format_array(train_generator.get_class_proportion()),
                        "augmult": train_generator.get_augmentation_ratio()
                    })

                    val_generator = SkincareDataGenerator(
                        images_df=dev_df,
                        images_dir=DATASET_IMG_DIR,
                        image_size=(image_size, image_size),
                        resize_filter=resize_filter,
                        color_space=color_space,
                        batch_size=batch_size,
                        image_augmentation=image_augmentation,
                        truth_df=dev_df,
                        truth_columns=columns_list,
                        shuffle=False,
                        is_multilabel=is_multilabel
                    )

                    out_dict.update({
                        "valsize": len(dev_df),
                        "valmalprop": format_array(val_generator.get_class_proportion()),
                    })

                    #
                    # Get the number of output classes from the generator
                    assert n_output_classes == train_generator.get_class_count()
                    assert n_output_classes == val_generator.get_class_count()

                    #
                    # Compute class weights
                    if class_weights == "default":
                        class_proportion = [1.0] * n_output_classes
                    elif class_weights == "compute":
                        class_proportion = train_generator.get_class_proportion()
                    else:
                        raise Exception(
                            "Column 'classweights' can be either 'default' or 'compute'. Found {}".format(class_weights))
                    print("Class Weights: Using class proportions {}".format(class_proportion))

                    #
                    # TRAINING
                    assert classifier is not None

                    with open(os.path.join(out_dirname, "{}-model_summary.txt".format(idx)), "w") as summary_file:
                        def writeln(s):
                            summary_file.write(s)
                            summary_file.write("\n")

                        classifier.get_model().summary(print_fn=writeln)

                    #
                    #
                    # DO THE REAL THING...
                    train_start_time = datetime.datetime.now()
                    out_dict.update({
                        "trainstart": train_start_time.strftime("%Y%m%d-%H%M%S")
                    })

                    print("Invoking train()...")
                    train_res = classifier.train(
                        out_dir=out_dirname,
                        train_generator=train_generator,
                        val_generator=val_generator,
                        epochs=train_epochs,
                        n_cpus=n_cpus,
                        class_freq_distr=class_proportion)

                    train_end_time = datetime.datetime.now()
                    training_time = train_end_time - train_start_time
                    #
                    #
                    #

                    # Compute accuracy
                    train_accuracy = train_res.last_model.history.history["acc"][0]

                    out_dict.update({
                        "traintime": str(training_time),
                        "trainacc": "{:.3f}".format(train_accuracy)
                    })

                    notes.append("Best train epoch: {}".format(train_res.best_epoch))

                    #
                    # Save the model
                    model_save_filepath, _ = train_res.save_best_model(out_dir=out_dirname, prefix="{}-".format(idx))
                    print("Saved best trained model to " + model_save_filepath)
                    _, model_save_filename = os.path.split(model_save_filepath)

                    out_dict.update({
                        "trainmodel": str(model_save_filename)
                    })

                    #
                    # Save the LOSS plot
                    generate_trainval_plot(train_res.iteration_history,
                                        os.path.join(out_dirname, "{}-plot-loss.png".format(idx)))
                    generate_accval_plot(train_res.iteration_history,
                                        os.path.join(out_dirname, "{}-plot-accuracy.png".format(idx)))

                    #
                    # VALIDATION
                    # Testing on the same dataset used for validation during training
                    # The test generator must not augment images and must not shuffle
                    dev_test_generator = SkincareDataGenerator(
                        images_df=dev_df,
                        images_dir=DATASET_IMG_DIR,
                        image_size=(image_size, image_size),
                        resize_filter=resize_filter,
                        color_space=color_space,
                        batch_size=batch_size,
                        image_augmentation="none",
                        truth_df=dev_df,
                        truth_columns=columns_list,
                        shuffle=False,
                        is_multilabel=is_multilabel
                    )

                    # Run test on the validation set
                    dev_test_pred = classifier.predict(generator=dev_test_generator)
                    dev_test_res = inspect_predictions(ground_truth=dev_df, predictions=dev_test_pred)

                    #
                    # Store data for validation
                    if not args.skip_roc:
                        generate_ROC_plot(dev_test_res, os.path.join(out_dirname, "{}-dev-ROC".format(idx)))

                    out_dict.update({
                        "valacc": dev_test_res.metrics["accuracy"],
                        "valspec": dev_test_res.metrics["specificity"],
                        "valsens": dev_test_res.metrics["sensitivity"]
                    })

                    #
                    # TEST
                    if test_df is not None:

                        # The test generator must not augment images and must not shuffle
                        test_generator = SkincareDataGenerator(
                            images_df=test_df,
                            images_dir=DATASET_IMG_DIR,
                            image_size=(image_size, image_size),
                            resize_filter=resize_filter,
                            color_space=color_space,
                            batch_size=batch_size,
                            image_augmentation="none",
                            truth_df=test_df,
                            truth_columns=columns_list,
                            shuffle=False,
                            is_multilabel=is_multilabel
                        )

                        # insert size data
                        out_dict.update({
                            "testsize": len(test_df),
                            "testmalprop": format_array(test_generator.get_class_proportion())
                        })

                        # Run test
                        test_pred = classifier.predict(generator=test_generator)
                        test_res = inspect_predictions(ground_truth=test_df, predictions=test_pred)

                        #
                        # Save test data
                        if not args.skip_roc:
                            generate_ROC_plot(test_res, os.path.join(out_dirname, "{}-test-ROC".format(idx)))

                        out_dict.update({
                            "testacc": test_res.metrics["accuracy"],
                            "testspec": test_res.metrics["specificity"],
                            "testsens": test_res.metrics["sensitivity"],
                            "testauc": format_array(test_res.auc)
                        })

                #
                # Catch all possible exceptions here, and go to next row.
                except Exception as e:
                    print("!!!EXCEPTION OCCURRED!!! Train {}: {}".format(idx, str(e)))
                    exceptions += str(e) + "\n"

        #
        # Free GPU memory
        print("Closing TF session...")
        tf_session.close()

        # Be sure that some of the intermediate model holders are swept away
        gc.collect()
        gc.collect()
        gc.collect()

        #
        # Fill the row of the output dataframe
        out_dict.update({
            "exceptions": exceptions,
            "notes": " ".join(notes),
        })

        # Check that we did NOT insert unexpected columns
        for outkey in out_dict.keys():
            assert outkey in DF_COLUMNS

        # Add a new row ti the output dataset
        print("Adding row:")
        print(str(out_dict)[:100])
        out_dicts.append(out_dict)

        #
        # Saving partial results
        row_df = pandas.DataFrame([out_dict], columns=DF_COLUMNS)
        row_df_filename = os.path.join(out_dirname, "{}-automation_result.csv".format(idx))
        print("Saving row automation results to " + row_df_filename)
        with open(row_df_filename, "w") as outfile:
            row_df.to_csv(outfile, index=False, header=True)

    #
    #
    out_df_filename = os.path.join(out_dirname, "automation_result.csv")
    print("Saving final automation results to " + out_df_filename)

    out_df = pandas.DataFrame(out_dicts, columns=DF_COLUMNS)
    # out_df.to_csv(sys.stdout, index=False, header=True)
    with open(out_df_filename, "w") as outfile:
        out_df.to_csv(outfile, index=False, header=True)

    print("All done.")
