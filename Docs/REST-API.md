# Skincare: REST API

The Skincare implementation provides a web REST API to perform classification, segmentation, and feature extraction.

The calls return either a JSON dictionary or an image.

In the following, the REST commands available from the classification server.

## Info

Returns information about the classifiers.

* path: `/model_info`
* method: GET
* input parameters: none
* returns: a JSON dictionary with information about the server.


Example:

```JSON
{
    "binary_classification_model": "../../GoodModels/0-keras_model-20190412-142934.h5",
    "binary_classification_classes": [
        "BEN",
        "MAL"
    ],
    "binary_classification_classes_description": [
        "benign",
        "malignant"
    ],
    "eight_class_classification_model": "/home/fnunnari/Documents/SkinCare/Code/GoodModels/Model-ISIC2019-8cls-450px-25k/0-keras_model-20190807-122631.h5",
    "eight_class_classification_classes": [
        "MEL",
        "NV",
        "BCC",
        "AK",
        "BKL",
        "DF",
        "VASC",
        "SCC"
    ],
    "eight_class_classification_classes_description": [
        "Melanoma",
        "Melanocytic nevus",
        "Basal cell carcinoma",
        "Actinic keratosis",
        "Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)",
        "Dermatofibroma",
        "Vascular lesion",
        "Squamous cell carcinoma"
    ],
    "segmentation_model": "../../GoodModels/model-segment-weights.h5",
    "feature_extraction_classes": [
        "globules",
        "streaks",
        "pigment_network",
        "milia_like_cyst",
        "negative_network"
    ]
}
```


## Static content

Used to serve static html page, useful to enrich the service.
Just create an `html` subdirectory in your server working directory and store your files there.

* path: `/html/<file>`
* method: POST
* input parameters: the name of the file to retrieve
* returns: the specified file, contained in the `html` subdirectory.
  * Content-Type: inferred from the file extension


## Classify Benign vs. Malignant

Binary classification is the process of analyzing an image of a lesion and return the probability distribution between _benign_ (0) and _malignant_ (1).

* path: `/classify/binary`
* method: POST
* input parameters:
  * `file`: the image to classify. Must be a JPEG or PNG image. No alpha.
* returns: a JSON structure with info about the classification with the following fields:
  * `error` If there was an error, otherwise this entry is absent
  * `filename` The name of the file provided as input
  * `prediction` A 2-dimension array with the probability for benign and malignant cases, respectively.
  * `confidence` A 2-dimension array with the confidence level, in a range [-1.0,1.0].
    Only predictions with positive confidence should be taken into account.

## Classify 8 classes

Eight-level classification is the process of analyzing an image of a lesion and return the probability distribution
 between the following eight classes: Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis,
 Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis), Dermatofibroma,
 Vascular lesion, Squamous cell carcinoma.

* path: `/classify/eight_class`
* method: POST
* input parameters:
  * `file`: the image to classify. Must be a JPEG or PNG image. No alpha.
* returns: a JSON structure with info about the classification with the following fields:
  * `error` If there was an error, otherwise this entry is absent
  * `filename` The name of the file provided as input
  * `prediction` A 8-dimension array with the probability for benign and malignant cases, respectively.
    For example [0.1,0.9] means that it is very likely a malignant case.
  * `confidence` A 8-dimension array with the confidence level, in a range [-1.0,1.0].
    Only predictions with positive confidence should be taken into account.


## Segmentation

Segmentation is the process of taking the image of a lesion as input and generating another image representing a binary _mask_ containing the lesion.

The output image is a greyscale PNG image, pixels can be only black (the pixels are outside of the lesion) or white (the pixels pertains to the lesion).

* path: /segment
* method: POST
* input parameters:
  * `file`: the image to segment. Must be a JPEG or a PNG image. No alpha.
* returns:
  * In case or error, returns a JSON file with the error reason:
    * Content-Type: `application/JSON`
    * `error`: The reason of the error
  * Otherwise, returns a PNG image (same size as input), greyscale (1 channel), with the lesion mask.
    * Content-Type: `image/png`

## Feature Extraction

Feature Extraction is the process of extracting, from the image of a lesion, pixel areas classified as pertaining to a certain category.

The output image is a greyscale PNG image, pixels can be only black (the pixels are not in the feature class) or white (the pixels pertains to the feature class).

* path: /extract_feature/<feature_classs>
  * with `feature_class` one among: `globules`, `streaks`, `pigment_network`, `milia_like_cyst`, `negative_network`.
* method: POST
* input parameters:
  * `file`: the image to segment. Must be a JPEG or a PNG image. No alpha.
* returns:
  * In case or error, returns a JSON file with the error reason:
    * Content-Type: `application/JSON`
    * `error`: The reason of the error
  * Otherwise, returns a PNG image (same size as input), greyscale (1 channel), with the lesion mask.
    * Content-Type: `image/png`


## SendImage Python script

If you need to test the server API calls that require an image as input, you can use the following script.

```python
import sys
import os
import requests

import pprint

"""
E.g.:
python SendImage.py ../../../DataSets/ISIC/ISIC-190110/Images/ISIC_0000000.jpeg http://127.0.0.1:5000/classify/binary

"""


if len(sys.argv) < 3:
    print("Usage: SendImage <filename:str> <url:str>")
    print("If the answer is an application/json, it will be printed in the standard output.")
    print("If the answer is an image/'extension', it will be save as '<filename>-answer.<extension>'.")
    exit(10)

image_filepath = sys.argv[1]
destination_url = sys.argv[2]

print("Sending image {} to url {}".format(image_filepath, destination_url))

if not os.path.exists(image_filepath):
    raise Exception("Path {} doesn't exist".format(image_filepath))

if not os.path.isfile(image_filepath):
    raise Exception("Path {} is not a file".format(image_filepath))

_, filename = os.path.split(image_filepath)
filename_root, filename_extension = os.path.splitext(filename)
content_type = 'image/' + filename_extension
files = {'file': (filename, open(image_filepath, 'rb'), content_type)}

response = requests.post(url=destination_url, files=files)

print("Answer code: {}".format(response.status_code))

if response.status_code == requests.codes.ok:

    print("Got response status OK")
    response_content_type = response.headers["Content-Type"]
    print(response_content_type)

    #
    # For images
    if response_content_type.startswith("image/"):
        answer_img_extension = response_content_type[len("image/"):]
        save_image_filename = filename_root + "-answer." + answer_img_extension
        print("Saving to " + save_image_filename)
        with open(save_image_filename, 'wb') as f:
            f.write(response.content)
        pass

    #
    # For JSON content
    elif response_content_type.startswith("application/json"):
        pprint.pprint(response.json())

    #
    # For plain HTML content
    elif response_content_type.startswith("text/html"):
        print(response.content)

    #
    # If we don't know what to do with it
    else:
        print("Unrecognized content type. Passing.")

else:
    print("There was an HTTP error: {}".format(response.reason))


print("Done.")
```

Output Example:

```JSON
{
    "filename": "ISIC_0000001.jpeg",
    "prediction": "[0.82298684 0.17701308]"
}
```

Meaning that the image is with 82% probability benign.

Or, in case of wrong file type:

```JSON
{
    "error": "cannot identify image file '/tmp/tmpvo2qvnh5/readme.txt'",
    "filename": "readme.txt"
}
```
