# Skincare: REST API

The Skincare implementation provides a web REST API to perform classification, segmentation, and feature extraction.

The calls return either a JSON dictionary or an image.

In the following, the REST commands available from the classification server.

In this document, we _assume_ an empty string as REST prefix in the server configuration file:

    "REST_API_URL_PREFIX": "",

## Info

Returns information about the classifiers.

* path: `/info`
* method: GET
* input parameters: none
* returns: a JSON dictionary with information about the server.

Example:

```json
{
  "classification": {
    "model_file": "../Model-ISIC2019-8cls-VGG16flat-227px-20k/0-keras_model.h5"
  }
}
```

## Classification

Classification is the process of analyzing an image of a lesion
and returning the probability distribution among the classes.

* path: `/classify`
* method: POST
* input parameters:
  * `file`: the image to classify. Must be a JPEG or PNG image. No alpha.
* returns: a JSON structure with info about the classification with the following fields:
  * `error` If there was an error, otherwise this entry is absent
  * `filename` The name of the file provided as input
  * `prediction` An array with the probability distribution. By definition, elements sum to 1.0.
  * `confidence` An array with the confidence level, in a range [-1.0,1.0].
    Only predictions with positive confidence should be taken into account.

## Visual Explanation of classification

* path: `/explain/<method>/<layer>`
* method: POST
* input parameters:
  * `file`: the image to classify. Must be a JPEG or PNG image. No alpha.
  * `method`: is the visual explanation method. At the moment either `gradcam` for Grad-CAM, or `rise` for RISE.
* returns: a JSON structure including Base64-encoded images for visual explanation:
  * `error` If there was an error, otherwise this entry is absent
  * `filename` The name of the file provided as input
  * `method` The explanation method provided as input
  * `layer_name` The name of the layer passed as input
  * `grey`: A Base64-encoded greyscale PNG image of the saliency map extracted by the XAI algorithm. The resolution is the same as of the original image. 
  * `heatmap`: A Base64-encoded color RGB PNG image: the colorful version of the greymap.
  * `comp`: A Base64-encoded color RGB PNG image with a composition between the original input image and the colored heatmap.

## Segmentation (IN PROGRESS)

Segmentation is the process of taking an image as input and generating another image representing a binary _mask_ highlighting regions of interest.

The output image is a greyscale PNG image, pixels can be only black (the pixels are outside of the regions of interest) or white (the pixels pertains to the regions of interest).

* path: `/segment`
* method: POST
* input parameters:
  * `file`: the image to segment. Must be a JPEG or a PNG image. No alpha.
* returns:
  * In case or error, returns a JSON file with the error reason:
    * Content-Type: `application/JSON`
    * `error`: The reason of the error
  * Otherwise, returns a PNG image (same size as input), greyscale (1 channel), with the lesion mask.
    * Content-Type: `image/png`
