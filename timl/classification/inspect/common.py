import numpy as np
from pandas import DataFrame

from typing import Dict
from typing import Union


# The columns used for the per-image prediction report
PER_IMAGE_COLUMNS = ["image", "groundtruth", "class_groundtruth", "pdist", "inference", "class_inference", "correct"]


class ClassificationResults:
    def __init__(self,
                 ground_truth: DataFrame,
                 predictions: np.ndarray,
                 per_image_results: DataFrame,
                 summary: Dict[str, Union[str, float]]
                 ):
        """
        :param predictions: predictions numpy array.
        :param ground_truth: the ground truth dataframe. #rows==n_samples, #columns==image_name + n_classes
        :param per_image_results: summary result per each image. Same order of images as in the predictions.
         #rows==n_samples. Columns are the images name followed by ground truth, predictions,
          and class-level information, including correctness.
        :param summary: A dictionary mapping information or metric results into their respective value.
        """

        self.predictions = predictions
        self.ground_truth = ground_truth
        self.images_info = per_image_results
        self.metrics = summary
        self.auc = []


def inspect_predictions(ground_truth: DataFrame, predictions: np.ndarray) -> ClassificationResults:
    """

    :param ground_truth: The ground truth as pandas Dataframe, as downloaded from the ISIC challege:
     1 column with image code, followed by the columns of ground truth in 1-hot format.
    :param predictions: a numpy.ndarray with the predictions, as normally returned by a softmax.
    Must have the same number of rows as the ground_truth. Must contain only the predicted p-distribution.
    :return: An instance of Classification Results
    """

    from timl.common.utils import format_array
    from timl.common.datageneration import IMAGES_COLUMN

    classes = ground_truth.columns[1:]
    n_classes = len(classes)
    n_samples = len(ground_truth)
    image_names = ground_truth[IMAGES_COLUMN]

    #
    # Some consistency checks
    if len(classes) != predictions.shape[1]:
        raise Exception("Number of columns is different: {} vs. {}.".format(len(classes), predictions.shape[1]))

    if n_samples != predictions.shape[0]:
        raise Exception("Number of samples is different: {} vs. {}.".format(n_samples, predictions.shape[0]))

    # Get the p distribution
    assert predictions.shape[0] == n_samples
    assert predictions.shape[1] == n_classes
    # get the index of the class with highest prediction
    y_pred_cls = np.argmax(predictions, axis=1)

    # Convert to 1-hot
    # numpy.eye(number of classes)[vector containing the labels]
    y_pred = np.eye(n_classes)[y_pred_cls]
    y_pred_bool = y_pred.astype(dtype=bool)

    # Extract the ground truth (without image_names)
    y_truth_df = ground_truth[classes]
    # The truth would be in 1-hot format
    # Get it as boolean
    y_truth_bool = y_truth_df.astype(dtype=bool)
    y_truth_cls = np.argmax(np.asarray(y_truth_df), axis=1)

    # We must have the same number of samples for both the prediction and the ground truth
    assert len(y_pred) == len(y_truth_df)

    y_correct = (y_pred_cls == y_truth_cls)

    #
    # Compose by-image output dictionary
    #
    # Remove absolute path and leave only the file name
    truths = [format_array(a[1:], format="{:.0f}") for a in y_truth_df.itertuples()]
    assert len(truths) == n_samples
    p_dists = [format_array(img_ps) for img_ps in predictions]
    assert len(p_dists) == n_samples
    inferences = [format_array(inf, format="{:.0f}") for inf in y_pred]
    assert len(inferences) == n_samples
    class_inferences = y_pred_cls
    assert len(class_inferences) == n_samples

    images_info = {"image": image_names,
                   "groundtruth": truths,
                   "class_groundtruth": y_truth_cls,
                   "pdist":  p_dists,
                   "inference": inferences,
                   "class_inference": class_inferences,
                   "correct": y_correct
                   }

    # Check that all columns are there
    for col in PER_IMAGE_COLUMNS:
        assert col in images_info.keys()

    images_info_df = DataFrame.from_dict(data=images_info)
    images_info_df = images_info_df.reindex(columns=PER_IMAGE_COLUMNS)

    #
    # Compute specificity and sensitivity for multi-class case
    # https://stats.stackexchange.com/questions/321041/how-to-find-sensitivity-and-specificity-for-more-than-two-levels-in-the-dependen

    #
    # Compute per-class stats
    #
    P = np.sum(y_truth_bool)
    N = np.sum(~y_truth_bool)
    TP = np.sum(y_pred_bool & y_truth_bool)
    TN = np.sum((~y_pred_bool) & (~y_truth_bool))
    FP = np.sum(y_pred_bool & (~y_truth_bool))
    FN = np.sum((~y_pred_bool) & y_truth_bool)

    accuracy = (TP + TN) / (P + N)
    specificity = TN / N
    sensitivity = TP / P
    f1 = 2 * TP / ((2 * TP) + FP + FN)
    # TODO -- add precision

    # Class proportions
    freq_count = y_truth_df.sum(axis=0)
    freq_count = freq_count / n_samples
    cls_props = freq_count.to_numpy().tolist()

    #
    # Compute intra-class stats
    #
    summary = {
        "samples": n_samples,
        "classes": ';'.join([str(c) for c in classes]),
        "class_prop": format_array(cls_props),
        "accuracy": format_array(accuracy),
        "specificity": format_array(specificity),
        "sensitivity": format_array(sensitivity),
        "f1": format_array(f1),
        "global_accuracy": np.sum(y_correct) / n_samples,
        "mean_accuracy": np.sum(accuracy) / n_classes,
        "weighted_mean_accuracy": np.sum(cls_props * accuracy),
        "mean_specificity": np.sum(specificity) / n_classes,
        "weighted_mean_specificity": np.sum(cls_props * specificity),
        "mean_sensitivity": np.sum(sensitivity) / n_classes,
        "weighted_mean_sensitivity": np.sum(cls_props * sensitivity),
        "mean_f1": np.sum(f1) / n_classes,
        "weighted_mean_f1": np.sum(cls_props * f1)
    }

    #
    # TODO - Confusion matrix

    #
    # TODO - Prediction with different thresholds.
    #    out2_df=pandas.DataFrame(classifier.eval_metricvsthreshold_table())
    #    out2_filename = os.path.join(out_dirname, "metricsvsthreshold_table.csv")
    #    with open(out2_filename, "w") as outfile:
    #        out2_df.to_csv(outfile, index=False, header=True)

    return ClassificationResults(predictions=predictions, ground_truth=y_truth_df,
                                 per_image_results=images_info_df,
                                 summary=summary
                                 )


#
# TODO
#
# Code to analyze performances on different thresholds.
# Not used, not tested, yet
def eval_metricvsthreshold_table(self, Thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    TP =[None]*len(Thresholds)
    TN =[None]*len(Thresholds)
    Accuracy =[None]*len(Thresholds)
    Specificity =[None]*len(Thresholds)
    Sensitivity =[None]*len(Thresholds)
    YoudenIndex =[None]*len(Thresholds)

    gt = np.asarray(self.gt, dtype=bool)
    P = np.sum(gt)
    N = np.sum(~gt)

    for i,thr in enumerate(Thresholds):
        inference = self.score[:,1]>thr
        TP[i] = np.sum(inference & gt)
        TN[i] = np.sum((~inference) & (~gt))

        Accuracy[i] = (TP[i] + TN[i]) / (P + N)
        Specificity[i] = TN[i] / N
        Sensitivity[i] = TP[i] / P
        YoudenIndex[i] = Specificity[i] + Sensitivity[i] - 1

    return {"Thresholds": Thresholds,
             "Accuracy": Accuracy,
             "Specificity":Specificity,
             "Sensitivity":Sensitivity,
             "YoudenIndex":YoudenIndex}
