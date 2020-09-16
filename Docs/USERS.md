# TIML - Users Manual

## Training class prediction with automation

We implemented an automated training procedure.

1. Prepare an input CSV file. A sample is provided in file `data/automation_test_input_1.csv`. In this file, each row specifies a training model and several parameters.
2. Execute the automation module providing the CSV input file.
3. Wait
4. The output will be another in a directory called `skincare_train_output-YYYYMMDD-hhmmss` containing:
    * The overall `automation_result.csv` CSV file with the remaining columns filled with the computed data.
    * For each row in the input file, the trained model will be saved as `N-keras_model-YYMMDD-hhmmss.h5`, where N is the row number of the trained model.
    * For each row the plots of loss function and ROC graphs.
    * For each row the training results, saved for possibly later analysis of the loss evolution and re-plotting.

Example:

```bash
cd Classifiers
# A short example, only one line.
python -m skincare.automation ../data/automation_test_input_1.csv
# A longer example, more lines and some augmentation. Might take a while.
python -m skincare.automation ../data/automation_test_input_2.csv
```

Options can be passed to the automation command line:

```txt
$ python -m skincare.automation -h
Searching for skincare_config.json
Skincare configuration loaded.
usage: __main__.py [-h] [--img-dir IMG_DIR] [--cuda-gpu CUDA_GPU] <input_table.csv>

Automated training and testing of CNNs for multi-class prediction.

positional arguments:
  <input_table.csv>    The CSV table with the input information.

optional arguments:
  -h, --help           show this help message and exit
  --img-dir IMG_DIR    The directory path in which to look for the images. If
                       omitted, uses the one specified in skincare_config.json
  --cuda-gpu CUDA_GPU  The CUDA GPU number to use for training
```


### Automation input

The automation input CSV file must have a value for the following columns (the remaining columns will be filled by the code):

* `method` The configuration fo the training. It corresponds to a network plus a set of fixed hyper-parameters:
  * `VGG16` A VGG16 CNN keeping the weight of a training on imagenet (Transfer learning). The last two layers are set to 2048 and 2048 and initialized randomly. Uses SGD optimizer.
  * `VGG16_Nadam` Same as before but uses NAdam optimizer.
  * `VGG16_Adadelta` Same as before but uses AdaDelta optimizer.
  * `SC19` Custom lightweight version derived from AlexNet. No Transfer.

* `dataset` The dataset used for training/velidation/testing (validation test is also known as _development_ set).
* `split` The strategy used to split the dataset in train/validation/test
  * `pre` Precomputed and stored on disk. We take the name specified in `dataset` (e.g. `ISIC-190213`) and look for files named `ISIC-190213-train.csv`, `ISIC-190213-dev.csv`, and `ISIC-190213-test.csv`.
  * `frac=<proportion:float>` A float number `frac > 0 and frac < 0.5`, indicating the proportion to sample. For a dataset of size 1000 elements, `frac=0.1` means that 100 elements will be extracted for dev and another 100 for test, leaving the train size at 800 samples.
  * `n=<n_samples:int>` An integer number indicating how many samples to extract for velidation and for test. For a dataset of size 1000 elements, `n=100` means that 100 elements will be extracted for dev and another 100 for test, leaving the train size at 800 samples.
  * `list` The `dataset` is read as a semi-colon separated list of files. They can be either 2 (train/val) or 3 (train/val/test) files E.g.:, `data/train.csv;data/val.csv`.
* `epochs` Number of epochs for training.
* `imgaug` Image augmentation preset:
  * `none` No augmentation
  * `hflip` Each image is also flipped horizontally
  * `hflip_rot24` Each image is also flipped horizontally and rotated 24 times (15deg steps).
  * `hflip_rot4` Each image is also flipped horizontally and rotated 4 times (90deg steps). Avoids black corners.
* `batchsize` batchsize for training.
* `imgsize` Images are rescaled to this resolution (square) before augmentation.
* `resizefilter` The filter used to resize the images. Can be:
  * `nearest`
  * `bilinear`
  * `bicubic`
  * `lanczos` 
* `colorspace` Specifies weather to keep the original RGB colorspace or convert the image to another format. PIL is used for mode conversion: <https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#concept-modes>.
  * `RGB` It is supposed to be the default when loading jpg images.
  * `HSV` Uses the Hue, Saturation, Value colorspace. See: <https://en.wikipedia.org/wiki/HSL_and_HSV>.
  * `LAB` Uses the L\*a\*b colorspace. See: <https://en.wikipedia.org/wiki/CIELAB_color_space>.
  * `YCbCr` See: <https://en.wikipedia.org/wiki/YCbCr>
  * TODO: `CMYK` (Will require dynamic image depth). See: <https://en.wikipedia.org/wiki/CMYK_color_model>.   
* `classcolumns`
  * `default` From the training dataset, all the columns except the first (`image_name`) will be used.
  * If a semi-column-separated list (e.g., `BEN;MAL`), this is the list of columns where each line is the one-hot vector representing the ground truth classification.
  * If a single string (e.g., `ben_mal`) the name of a single column containing the classes as factors. The total number of classes will be automatically computed. Factors will se sorted alphabetically.
* `classweights` The weight distribution of the classes, normally to compensate uneven counts in the training set.
  * `default` All classes are left to the default 1.0 weight.
  * `compute` analyzes the dataset and automatically set the weight according to the frequency of appearance of the class in the dataset.

## Predicting classes

The `predict` module allows you to generate multiclass predictions
(the probability distribution, generally an output of the softmax layer)

```
python -m skincare.classification.predict --help
Using TensorFlow backend.
Searching for skincare_config.json
Skincare configuration loaded.
usage: __main__.py [-h] [--img-dir IMG_DIR] [--cuda-gpu CUDA_GPU]
                   <keras_model.h5> <config_table.csv> <test_dataframe.csv>
                   <output.csv>
```

Example: 

    python -m skincare.classification.predict models/Model-ISIC2019-8cls-450px-20k/0-keras_model-20190803-075631.h5 models/Model-ISIC2019-8cls-450px-20k/0-automation_result.csv ../data/ISIC_Challenge_2019/ISIC_2019_Training_GroundTruth-nounk-20k-test.csv isic2019-prediction.csv --img-dir=/mnt/XMG4THD/skincaredata/ISIC_Challenge_2019/ISIC_2019_Training_Input/

Will generate:

* `prediction-train-20k.csv` containing the predictions (and inferred class) in text format.
* `prediction-train-20k.npy` containing the prediction as numpy array
* `prediction-train-20k/`    directory with the predictions, one file per image
  * File format: `image_name-0000000000.npy`.
  * Why? It is the same format used by the system caching the activation values. So that the output predictions can be read back from the same DataGenerator. This has been used when appending metadata to the final predictions to improve accuracy.

## Testing class prediction

Models can be tested on any dataset using the `inspect` submodule.

Help is available:

```
python -m skincare.classification.inspect --help
usage: __main__.py [-h] [--overwrite] <test_dataframe.csv> <predictions.csv>

Automated training and testing of CNNs for multi-class prediction.

positional arguments:
  <test_dataframe.csv>  The CSV table with the ground_truth. First column is
                        the image_name, followed by the predicted classes.It
                        will be used as dictionary: the image name is used to
                        extract the ground truth of the images in the
                        predictions.csv
  <predictions.csv>     The CSV containing the predictions. First column is
                        the image_name, followed by the predicted classes.It
                        is the format generated by
                        skincare.classification.predict.__main__

optional arguments:
  -h, --help            show this help message and exit
  --overwrite           If true, overwrites data inside the destination
                        directory.
```

Example:

    python -m skincare.classification.inspect ../data/ISIC_Challenge_2019/ISIC_2019_Training_GroundTruth-nounk-20k-test.csv isic2019-prediction.csv

It will create a directory by concatenating the name of the test set and the name of the model, like:

    inspect_ISIC_2019_Training_GroundTruth-nounk-20k-test-isic2019-prediction/

In this directory you will find:
* `summary.csv` containing all the metrics of the test
  * Also in JSON format as `summary.json`
* `per_image_results.csv` containing predictions and results for each image
* a set of `ROC-xx-cls.png` image files, with the ROC plot.
