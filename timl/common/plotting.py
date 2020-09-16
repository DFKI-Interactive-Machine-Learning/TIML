from timl.classification.classifier import TrainHistory
from timl.classification.inspect.common import ClassificationResults
import numpy as np
from matplotlib import pyplot as plt


def generate_trainval_plot(itrhistory: TrainHistory, savefilename: str) -> None:

    plt.figure()

    # Performs a convolution on the train data to soften the plot
    avg_over = 6
    avg_filter = np.ones(avg_over)/avg_over
    # Extend the array to have extra margin for convolution at edges.
    a = np.pad(itrhistory.losses, (avg_over//2, avg_over//2), 'reflect')
    b = np.convolve(a, avg_filter, mode='valid')
    #print(len(itrhistory.losses), a.shape, b.shape)
    #print(x)
    plt.plot(b)

    # Plots the validation data
    x = np.linspace(0, len(itrhistory.losses), num=len(itrhistory.val_losses)+1)
    plt.plot(x[1:], itrhistory.val_losses)

    # Plot ticks and labels
    plt.xticks(x, np.arange(len(x)))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'val'], loc='upper right')

    plt.savefig(savefilename)
    # Removing close, trying to fix the problem with `Tcl_AsyncDelete: async handler deleted by the wrong thread`
    # See: https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
    # plt.close()


def generate_accval_plot(itrhistory: TrainHistory, savefilename: str) -> None:

    plt.figure()
    avg_over = 6

    x = np.linspace(0, len(itrhistory.accuracies), num=len(itrhistory.val_accuracies)+1)
    avg_filter = np.ones(avg_over)/avg_over

    a = np.pad(itrhistory.accuracies, (avg_over//2, avg_over//2), 'reflect')

    b = np.convolve(a, avg_filter, mode='valid')
    print(len(itrhistory.accuracies), a.shape, b.shape)
    print(x)
    plt.plot(b)
    plt.plot(x[1:], itrhistory.val_accuracies)

    plt.xticks(x, np.arange(len(x)))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'], loc='upper right')

    plt.savefig(savefilename)
    # Removing close, trying to fix the problem with `Tcl_AsyncDelete: async handler deleted by the wrong thread`
    # See: https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
    # plt.close()


def generate_ROC_plot(data: ClassificationResults, savefilename_prefix: str) -> None:

    data.auc = []

    # n_samples = len(data.ground_truth)
    assert data.predictions.shape[0] == len(data.ground_truth)

    for i, col in enumerate(data.ground_truth.columns):
        print("Computing ROC and plot for class {}".format(col))

        # Take the predictions and ground truth of the class i
        p = data.predictions[:, i]
        y = data.ground_truth[col]

        unp, idp = np.unique(p, return_inverse=True)

        Tp = np.sum(y == 1)
        Tn = np.sum(y == 0)
        tp = 0
        fp = 0
        tpr = [0]
        fpr = [0]
        auc = 0
        for n in range(len(unp)-1, -1, -1):
            tmp = y[idp == n]
            dtp = np.sum(tmp == 1)
            dfp = np.sum(tmp == 0)
            auc += (tpr[-1] + dtp/(2*Tp))*(dfp/Tn)
            tp += dtp
            fp += dfp
            tpr.append(tp/Tp)
            fpr.append(fp/Tn)

        # Store the computed auc back into the classification results
        data.auc.append(auc)

        # Plot init
        plt.figure()
        # Plot the reference (pure chance) diagonal line
        plt.plot([0, 1], color=((0.5, 0.5, 0.5)), linestyle='dashed', label='_nolegend_')
        # Plot the curve
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")

        plt.title("Class {} - ROC".format(col))
        plt.legend(["AUC=%.3f" % auc], loc='lower right')
        plt.savefig(savefilename_prefix + "-{:04d}-{}.png".format(i, col))
        # plt.close()
