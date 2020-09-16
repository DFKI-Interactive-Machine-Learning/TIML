import os
import argparse
import json

import pandas

from timl.common.utils import format_array
from timl.common.plotting import generate_ROC_plot

from .common import inspect_predictions
from timl.common.datageneration import IMAGES_COLUMN

#
# MAIN
#
if __name__ == "__main__":

    # E.g.: skincare_train_output-20190712-162535-BEN_MAL-227/0-keras_model-20190712-162535.h5 skincare_train_output-20190712-162535-BEN_MAL-227/0-automation_result.csv ../data/ISIC-dataframe-190213-simplified-test-10lines.csv
    args_parser = argparse.ArgumentParser(
        description='Automated training and testing of CNNs for multi-class prediction.')

    args_parser.add_argument('test_csv', metavar='<test_dataframe.csv>', type=str,
                             help="The CSV table with the ground_truth. First column is the image_name, followed by the predicted classes."
                                  "It will be used as dictionary: the image name is used to extract the ground truth of the images in the predictions.csv")
    args_parser.add_argument('predictions_csv', metavar='<predictions.csv>', type=str,
                             help="The CSV containing the predictions. First column is the image_name, followed by the predicted classes."
                                  "It is the format generated by timl.classification.predict.__main__")
    args_parser.add_argument('--out-dir', type=str,
                             help="Specifies the output directory. Creates it if not existing")
    args_parser.add_argument('--overwrite', action="store_true",
                             help="If true, overwrites data inside the destination directory.")

    # This possibly stops the execution if the arguments are not correct
    args = args_parser.parse_args()

    #
    # Create the output directory
    # os.path.splitext(os.path.split(inference_set_filename)[1])[0]
    if args.out_dir is not None:
        out_dirname = args.out_dir
    else:
        out_dirname = "inspect_{}-{}".format(os.path.splitext(os.path.split(args.predictions_csv)[1])[0],
                                             os.path.splitext(os.path.split(args.test_csv)[1])[0]
                                             )
    print("Writing Inspection results to '{}'".format(out_dirname))

    if os.path.exists(out_dirname):
        if not args.overwrite:
            raise Exception("Directory '{}' already exists. Use option --overwrite to force writing.".format(out_dirname))
    else:
        os.makedirs(out_dirname)

    #
    # Load input tables
    predictions_df = pandas.read_csv(filepath_or_buffer=args.predictions_csv)  # type: pandas.DataFrame
    truth_df = pandas.read_csv(filepath_or_buffer=args.test_csv)

    #
    # Verify that they have the same image names
    n_predictions = len(predictions_df)
    n_truths = len(truth_df)
    if n_predictions != n_truths:
        raise Exception("The two datasets are supposed to have the same size."
                        " Found #prediction={} and #selected_truths={}".format(n_predictions, n_truths))

    prediction_images = predictions_df[IMAGES_COLUMN]
    truth_images = truth_df[IMAGES_COLUMN]
    equals = prediction_images == truth_images  # type: pandas.Series
    # print("Type:", type(equals))
    # print(equals)
    # print("All: ", equals.all())
    if not equals.all():
        raise Exception("The list of image names differ between the two datasets")

    #
    # Inspect the predictions
    predictions_array = predictions_df.values[:, 1:]
    inspection_res = inspect_predictions(predictions=predictions_array, ground_truth=truth_df)
    #
    #

    #
    # Generate ROC
    roc_filename = os.path.join(out_dirname, "ROC")
    generate_ROC_plot(inspection_res, roc_filename)
    # This also inserted ROC values into the results
    inspection_res.metrics.update({'auc': format_array(inspection_res.auc)})

    #
    # Save the summary as JSON
    summary_filename = os.path.join(out_dirname, "summary.json")
    with open(summary_filename, "w") as summary_outfile:
        json.dump(obj=inspection_res.metrics, fp=summary_outfile, indent=4)

    #
    # Save the summary as CSV
    metrics_df = pandas.DataFrame(columns=inspection_res.metrics.keys())
    metrics_df = metrics_df. append(inspection_res.metrics, ignore_index=True)
    with open(os.path.join(out_dirname, "summary.csv"), "w") as summary_csv_outfile:
        metrics_df.to_csv(path_or_buf=summary_csv_outfile, header=True, index=False)

    #
    # Save per-image prediction results
    #
    per_image_filename = os.path.join(out_dirname, "per_image_results.csv")
    with open(per_image_filename, "w") as images_outfile:
        inspection_res.images_info.to_csv(images_outfile, index=False, header=True)

    print("All done.")
