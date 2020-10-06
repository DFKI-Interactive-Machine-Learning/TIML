# TIML - Users Manual

## Training a class (or label) prediction model

We implemented an automated training procedure.
It takes as input a CSV file where each line is a model to be trained and columns are the training parameters. 



1. Prepare an input CSV file. A sample is provided in file `Examples/Example01/sample_train_input_1.csv`.
   In this file, each row specifies a training model and the columns the training parameters.
2. Execute the automation module providing the CSV input file.
   Check for syntax with `python -m timl.classification.train --help`.
3. Wait...
4. The output will be another in a directory called `train_output-YYYYMMDD-hhmmss` containing:
    * The overall `automation_result.csv` CSV file. It copies the input CSV, plus the remaining columns filled with the training statistics data.
    * For each row in the input file, the trained model will be saved as `N-keras_model-YYMMDD-hhmmss.h5`, where N is the row number of the trained model.
    * For each row, the plots of loss function and ROC graphs.
    * For each row, the training results, saved for possibly later analysis of the loss evolution and re-plotting.

Example:

```bash
cd path/to/TIML
# A short example, only one line.
python -m timl.classification.train --input-table=sample_train_input_1.csv --img-dir=../../timl/data/ISIC2019/images --out-dir=train_output
```


### Train input parameters

Here the list of possible values for training input parameters:

* `method` The configuration for the training. It corresponds to a network plus a set of fixed hyper-parameters:
  * `VGG16` A VGG16 CNN keeping the weight of a training on imagenet (Transfer learning). The last two layers are set to 2048 and 2048 and initialized randomly. Uses SGD optimizer.
  * `VGG16-fc2k-ml` A VGG16 with 2X fully connected layers at 2048 nodes, for multi-label classification (sigmoid output)
  * `RESNET50` A RESNET50 classification architecture.
  * `RESNET50-fc4k-ml` A RESNET50 for multilabeling.
  * `DenseNet169-fc4k-ml` A DenseNet169 for multilabeling.
  
  The full list can be found in the package `timl/classification/classifier_factory.py` in function `make_classifier`.

* `dataset` The dataset used for training/validation/testing (validation test is also known as _development_ set).
  This is a CSV file where the first column is called `image_name` and the remaining columns contain the ground truth.
  An example is in `timl/data/ISIC2019/ISIC_2019_Training_GroundTruth_meta-train.csv`
* `split` The strategy used to split the dataset in train/validation/test
  * `pre` Precomputed and stored on disk. We take the name specified in `dataset` (e.g. `ISIC-190213`) and look for files named `ISIC-190213-train.csv`, `ISIC-190213-dev.csv`, and `ISIC-190213-test.csv`.
  * `frac=<proportion:float>` A float number `frac > 0 and frac < 0.5`, indicating the proportion to sample. For a dataset of size 1000 elements, `frac=0.1` means that 100 elements will be extracted for dev and another 100 for test, leaving the train size at 800 samples.
  * `n=<n_samples:int>` An integer number indicating how many samples to extract for validation and for test. For a dataset of size 1000 elements, `n=100` means that 100 elements will be extracted for dev and another 100 for test, leaving the train size at 800 samples.
  * `list` The `dataset` is read as a semi-colon separated list of files. They can be either 2 (train/val) or 3 (train/val/test) files E.g.:, `data/train.csv;data/val.csv`.
* `epochs` Number of epochs for training.
* `imgaug` Image augmentation preset:
  * `none` No augmentation
  * `hflip` Each image is also flipped horizontally
  * `hflip_rot24` Each image is also flipped horizontally and rotated 24 times (15deg steps).
  * `hflip_rot4` Each image is also flipped horizontally and rotated 4 times (90deg steps). Avoids black corners.
  
  The full list of augmentation methods can be found in package `timl/common/imageaugmentation.py` in function `image_provider_factory`.
* `batchsize` batchsize for training.
* `imgsize` Images are rescaled to this resolution (square) before augmentation. E.g., `224` means 224x224 pixels.
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
* `classweights` The weight distribution of the classes, normally to compensate unbalanced counts in the training set.
  * `default` All classes are left to the default 1.0 weight.
  * `compute` analyzes the dataset and automatically set the weight according to the frequency of appearance of the class in the dataset.

## Predicting classes/labels

The `timl.classification.predict` module allows you to generate multiclass predictions
(the probability distribution, generally an output of the softmax layer)



```
python -m timl.classification.predict --help

usage: __main__.py [-h] --keras-model <keras_model.h5> --config-table
                   <config_table.csv> --test-dataframe <test_dataframe.csv>
                   --img-dir IMG_DIR --output-csv <output.csv>
                   [--generate-numpy] [--cuda-gpu CUDA_GPU]

Given a list of images and a classification (or labeling) model, generates the
predictions in CSV and optionally as binary numpy files.

optional arguments:
  -h, --help            show this help message and exit
  --keras-model <keras_model.h5>
                        The keras model to use for testing.
  --config-table <config_table.csv>
                        The CSV table with the input used for training
                        (batch_size, imgfilter, colorfiler, classcolumns,...)
  --test-dataframe <test_dataframe.csv>
                        The CSV table with the list of images to test.
  --img-dir IMG_DIR     The directory path in which to look for the images.
  --output-csv <output.csv>
                        The CSV table with the list of images to test.
  --generate-numpy      In addition to the output.csv, also binary numpy
                        arrays will be saved. 1) One big numpy file
                        output.npy, 2) a directory output/ containing one .npy
                        array per sample.
  --cuda-gpu CUDA_GPU   The CUDA GPU number to use for computing predictions
```

Example: 

    python -m timl.classification.predict --keras-model=.../0-keras_model.h5 --config-table=.../0-automation_result.csv --test-dataframe=timl/data/ISIC2019/ISIC_2019_Training_GroundTruth_meta-test.csv --img-dir=timl/data/ISIC2019/images --output-csv=predictions.csv --generate-numpy

Will generate:

* `predictions.csv` containing the predictions (and inferred class) in CSV text format.
* `predictions.npy` containing the prediction as numpy array
* `predictions/`    directory with the predictions, one file per image
  * File format: `<image_name>-0000000000.npy`.
  * Why? It is the same format used by the system caching the activation values. So that the output predictions can be read back from the same DataGenerator. This can be used to feed prediction vectors to other trainings without re-running the convolution stage.

## Testing class prediction

Models can be tested on any dataset using the `timl.classification.inspect` module.

Help is available:

```
python -m timl.classification.inspect --help

usage: __main__.py [-h] --test-csv <test_dataframe.csv> --predictions-csv
                   <predictions.csv> [--out-dir OUT_DIR] [--overwrite]

Test predictions by comparing them with a ground truth.

optional arguments:
  -h, --help            show this help message and exit
  --test-csv <test_dataframe.csv>
                        The CSV table with the ground_truth. First column is
                        the image_name, followed by the predicted classes. It
                        will be used as dictionary: the image name is used to
                        extract the ground truth of the images in the
                        predictions.csv
  --predictions-csv <predictions.csv>
                        The CSV containing the predictions. First column is
                        the image_name, followed by the predicted classes. It
                        is the format generated by
                        timl.classification.predict.__main__
  --out-dir OUT_DIR     Specifies the output directory. Creates it if not
                        existing
  --overwrite           If true, overwrites data inside the destination
                        directory.
```

Example:

    python -m timl.classification.inspect --test-csv=timl/data/ISIC2019/ISIC_2019_Training_GroundTruth_meta-test.csv --predictions-csv=predictions.csv --overwrite --out-dir=inspection

If the output dir is not specified, it would create a directory by concatenating the name of the test set and the name of the model, like:

    inspect_ISIC_2019_Training_GroundTruth_meta-test-predictions/

In the output directory you will find:
* `summary.csv` containing all the metrics of the test
  * Also in JSON format as `summary.json`
* `per_image_results.csv` containing predictions and results for each image
* a set of `ROC-xx-cls.png` image files, with the ROC plot.

## Running the Web server

The http REST interface is implemented using Flask.

First, you need to create a file called `timl_server_config.json` in your home or working directory.
If it doesn't exist, the server will stop suggesting a sample configuration, e.g.:

```json
{
    "CLASSIFICATION_MODEL_description": "Path to the model file (.h5) for classification.",
    "CLASSIFICATION_MODEL": "../Model-ISIC2019-8cls-VGG16flat-227px-20k/0-keras_model.h5",
    "REST_API_URL_PREFIX_description": "The URL prefix to access the REST-API.",
    "REST_API_URL_PREFIX": "rest",
    "STATIC_PAGES_DIR_description": "Path to the directory containing the static files.",
    "STATIC_PAGES_DIR": "html",
    "STATIC_PAGES_URL_PREFIX_description": "The URL prefix to access the static pages.",
    "STATIC_PAGES_URL_PREFIX": "web"
}
```

To run the server from a terminal:

```bash
cd path/to/TIML
source timl-p3env/bin/activate
export FLASK_APP=timl.networking.__main__.py
python -m flask run
```

By default, the server takes connections on port 5000.
In order to specify a different port:

    python -m flask run --host=0.0.0.0 --port=80

Browse to the following addresses to check for life:

      http://localhost:5000/
      http://localhost:5000/rest/info
      http://localhost:5000/web/classify.html

### Using the REST-API

When you have trained a model, you can make it available via the web REST-API

See document [REST-API](Classifiers/REST-API.md) for the complete documentation.

For example, for classification, follow this:

```
cd TIML/Scripts
source ../../p3env-3.7/bin/activate
python SendImage.py ISIC_0000003.jpeg http://localhost:5000/rest/classify
Sending image /Users/fanu01-admin/Downloads/ISIC_0000003.jpeg to url http://localhost:5000/rest/classify
Answer code: 200
Got response status OK
application/json
{'confidence': [0.21843838691711426,
                0.636655330657959,
                -0.9998914153766236,
                -0.9998995327187004,
                -0.9866130789741874,
                -0.9995066245901398,
                -0.9997779080149485,
                -0.9999668421623937],
 'filename': 'ISIC_0000003.jpeg',
 'prediction': [0.316133588552475,
                0.6820734143257141,
                1.357307792204665e-05,
                1.2558410162455402e-05,
                0.0016733651282265782,
                6.167192623252049e-05,
                2.776149813144002e-05,
                4.1447297007835004e-06]}
Done.
```

In this answer, the `prediction` entry contains the output of the last layer: a softmax probability distribution.
The association between the index and the class name is depends on your training data.


For testing the extraction of saliency and heatmaps, follow this:

```
python SendImage.py ISIC_0000003.jpeg http://localhost:5000/rest/explain/gradcam/block5_conv3
Sending image ISIC_0000003.jpeg to url http://localhost:5000/rest/explain/gradcam/block5_conv3
Answer code: 200
Got response status OK
application/json
{'args': {'layer_name': 'block5_conv3'},
 'comp': 'iVBORw0KGg ... uQmCC',
 'grey': 'iVBORw0KGg ... QmCC',
 'heatmap': 'iVBORw0 ... 5CYII=',
 'layer': 'block5_conv3',
 'method': 'gradcam'}
Done.
```

The `grey`, `heatmap` and `comp` contain the Base64 encoding of PNG images showing the greyscale saliency map, the colored heatmap, and a composition between the original image and the saliency map, respectively.

Currently supported visual XAI methods are `gradcam` and `rise`.


### Web interface

IN PROGRES

To test the system open one of the following links in a browser:

* `http://127.0.0.1:5000/html/classify.html` Use the system through the desktop interface.
* `http://127.0.0.1:5000/html/classifytouch.html` Use the system through the desktop interface.
* `http://127.0.0.1:5000/html/evaluate.html` Evaluate classification results.
