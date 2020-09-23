import base64
import sys
import traceback

import timl.networking.config

from timl.classification.classifier import Classifier
from timl.segmentation.seg_unet import UNetSegmenter

import keras
import tensorflow as tf

from flask import Flask
from flask import request
from flask import Response, send_file, send_from_directory
from werkzeug.utils import secure_filename
from requests import post

import PIL.Image

import os
import json
import tempfile
from io import BytesIO

from typing import Optional
from typing import Dict


CLASSIFICATION_MODEL_FILE_PATH = timl.networking.config.get_server_classification_model_path()
# SEGMENTATION_WEIGHTS_PATH = timl.networking.config.get_segmentation_weights_path()

STATIC_HTML_DIR = timl.networking.config.get_server_static_dir()
STATIC_HTML_URL_PREFIX = timl.networking.config.get_server_static_url_prefix()
REST_URL_PREFIX = timl.networking.config.get_server_rest_url_prefix()

#
# All the following are global variables used to store the Keras models and avoid re-loading at every call.
# Loading must be done in the call invocation thread. Can not be done here in the main execution.
# All the model initializations and predictions must be done in a properly set Tensorflow Session and Graph.
# See here: https://github.com/keras-team/keras/issues/8538
# and here: https://github.com/keras-team/keras/issues/2397#issuecomment-385317242

#
# The binary classifier
# TODO -- rename, remove "binary"
binary_classification_session = None  # type: Optional[tf.Session]
binary_classification_classifier = None  # type: Optional[Classifier]

#
# The UNET segmenter
# Will be initialized at first usage
#segmenter = None  # type: Optional[UNetSegmenter]
#segmenter_session = None  # type: Optional[tf.Session]


#
# Check HTML dir presence
html_abs_path = os.path.join(os.getcwd(), STATIC_HTML_DIR)
if not os.path.exists(html_abs_path):
    print("Couldn't find dir '{}' for static html files.".format(html_abs_path))
    exit(10)

#
# Fix URL roots
if not REST_URL_PREFIX.startswith("/"):
    REST_URL_PREFIX = "/" + REST_URL_PREFIX
if not REST_URL_PREFIX.endswith("/"):
    REST_URL_PREFIX += "/"

if not STATIC_HTML_URL_PREFIX.startswith("/"):
    STATIC_HTML_URL_PREFIX = "/" + STATIC_HTML_URL_PREFIX
if not STATIC_HTML_URL_PREFIX.endswith("/"):
    STATIC_HTML_URL_PREFIX += "/"

print("The REST-API is accessible through the URL root '{}'".format(REST_URL_PREFIX))

print("Static html files will be searched in '{}' and accessible via URL root '{}'".
      format(html_abs_path, STATIC_HTML_URL_PREFIX))


# The Flask app, used to serve.
app = Flask(__name__)


#
# Routing functions
#

@app.route("/")
def home():
    return '<p>TIML Flask-based server running.</p>' \
           '<p>REST-API at root URL ' + REST_URL_PREFIX + '</p>'\
           '<p>Web pages at root URL ' + STATIC_HTML_URL_PREFIX + '</p>'\
           '<p>Check REST-API docs at project website for valid requests: ' \
           '<a href="https://github.com/DFKI-Interactive-Machine-Learning/TIML" target="_blank">https://github.com/DFKI-Interactive-Machine-Learning/TIML</a></p>'


#
# Static Pages forwarding
@app.route(STATIC_HTML_URL_PREFIX + '<path:filename>')
def static_html_loader(filename):
    return send_from_directory(directory=html_abs_path, filename=filename)

#
#
# REST-API
#
#


@app.route(REST_URL_PREFIX + 'ping')
def hello_world():
    return 'pong'


@app.route(REST_URL_PREFIX + 'echo/<myparameter>')
def echo(myparameter):
    return "You told me " + myparameter


@app.route(REST_URL_PREFIX + 'info')
def model_info():

    # TODO -- load the model(s) and give info about input-size, output classes, ...

    out = {
        "classification": {
            "model_file": CLASSIFICATION_MODEL_FILE_PATH,
        },
        # "segmentation": {
        #     "segmentation_model": SEGMENTATION_WEIGHTS_PATH,
        # }
    }

    return Response(json.dumps(out), mimetype="application/json")


@app.route(REST_URL_PREFIX + 'classify', methods=['POST'])
def classify():
    global binary_classification_session
    global binary_classification_classifier

    output = {}

    if "file" not in request.files:
        output["error"] = "Missing 'file'"
    else:

        file = request.files['file']
        filename = secure_filename(file.filename)
        output["filename"] = filename

        try:

            # Save the image into a temporary directory
            with tempfile.TemporaryDirectory() as tmpdirname:
                print('Created temporary directory', tmpdirname)

                img_full_tmp_path = os.path.join(tmpdirname, filename)

                file.save(img_full_tmp_path)

                # Initialize the binary classifier, if needed
                if binary_classification_classifier is None:

                    binary_classification_session = tf.Session(graph=tf.Graph())
                    with binary_classification_session.as_default():
                        with binary_classification_session.graph.as_default():
                            binary_classification_model = keras.models.load_model(filepath=CLASSIFICATION_MODEL_FILE_PATH)
                            binary_classification_classifier = Classifier()
                            binary_classification_classifier._model = binary_classification_model

                assert (binary_classification_classifier is not None) and (binary_classification_session is not None)

                with binary_classification_session.as_default():
                    with binary_classification_session.graph.as_default():
                        predicted_classes, confidences = binary_classification_classifier.predict_image(PIL.Image.open(img_full_tmp_path))

                output["prediction"] = predicted_classes
                output["confidence"] = confidences

        except Exception as ex:
            print(ex)
            output["error"] = str(ex)

    return Response(json.dumps(output), mimetype="application/json")


# @app.route(REST_URL_PREFIX + 'segment', methods=['POST'])
# def segment():
#     global segmenter
#     global segmenter_session
#
#     output = {}
#
#     if "file" not in request.files:
#         output["error"] = "Missing 'file'"
#     else:
#         file = request.files['file']
#         filename = secure_filename(file.filename)
#         output["filename"] = filename
#
#         try:
#
#             # Save the image into a temporary directory
#             with tempfile.TemporaryDirectory() as tmpdirname:
#                 print('Created temporary directory', tmpdirname)
#                 img_full_tmp_path = os.path.join(tmpdirname, filename)
#                 file.save(img_full_tmp_path)
#                 in_image = PIL.Image.open(fp=img_full_tmp_path)
#
#                 # Initialize the segmentation model, if needed
#                 if segmenter is None:
#                     segmenter_session = tf.Session(graph=tf.Graph())
#                     with segmenter_session.as_default():
#                         with segmenter_session.graph.as_default():
#                             # segmenter = UNetSegmenter(weights_file_path=SEGMENTATION_WEIGHTS_PATH)
#                             pass  # TODO -- re-introduce the UNet segmenter.
#
#                 with segmenter_session.as_default():
#                     with segmenter_session.graph.as_default():
#                         segmented_image = segmenter.segment_image(im=in_image)
#
#                 # Save the mask image into a temporary memory buffer and return it as image.
#                 mask_img_buf = BytesIO()
#                 segmented_image.save(fp=mask_img_buf, format='PNG')
#                 mask_img_buf.seek(0)
#                 return send_file(filename_or_fp=mask_img_buf, mimetype="image/png")
#
#         except Exception as ex:
#             print(ex)
#             output["error"] = str(ex)
#
#     return Response(json.dumps(output), mimetype='application/json')


@app.route(REST_URL_PREFIX + 'explain/<method>/<layer>', methods=['POST'])
def explain(method, layer):
    global binary_classification_session
    global binary_classification_classifier

    from timl.classification.classifier import Classifier

    output = {}

    if "file" not in request.files:
        output["error"] = "Missing 'file' parameter in request"
        return Response(json.dumps(output), mimetype="application/json")

    file = request.files['file']
    filename = secure_filename(file.filename)
    output["filename"] = filename

    try:

        print("Explain image classification, method={}, layer={}".format(method, layer))

        # Save the image into a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('Created temporary directory', tmpdirname)

            img_full_tmp_path = os.path.join(tmpdirname, filename)

            file.save(img_full_tmp_path)

            # Initialize the 8-class classifier, if needed
            if binary_classification_classifier is None:
                binary_classification_session = tf.Session(graph=tf.Graph())
                with binary_classification_session.as_default():
                    with binary_classification_session.graph.as_default():
                        binary_classification_model = keras.models.load_model(
                            filepath=CLASSIFICATION_MODEL_FILE_PATH)
                        binary_classification_classifier = Classifier()
                        binary_classification_classifier._model = binary_classification_model

            assert (binary_classification_classifier is not None) and (
                        binary_classification_session is not None)

            # Retrieve the model input dimensions
            w, h, _ = binary_classification_classifier.get_input_size()

            # extra_params = {'layer_name': 'block5_conv3'}
            extra_params = {'layer_name': layer}

            with binary_classification_session.as_default():
                with binary_classification_session.graph.as_default():
                    grey_map, heat_map, composite = \
                        binary_classification_classifier.generate_heatmap(
                            image=PIL.Image.open(img_full_tmp_path),
                            method=method,
                            **extra_params)

                    comp_buff = BytesIO()
                    composite.save(fp=comp_buff, format='PNG')
                    comp_buff.seek(0)
                    comp64 = base64.b64encode(comp_buff.getvalue()).decode("utf-8")

                    grey_buff = BytesIO()
                    grey_map.save(fp=grey_buff, format='PNG')
                    grey_buff.seek(0)
                    grey64 = base64.b64encode(grey_buff.getvalue()).decode("utf-8")

                    heat_buff = BytesIO()
                    heat_map.save(fp=heat_buff, format='PNG')
                    heat_buff.seek(0)
                    heat64 = base64.b64encode(heat_buff.getvalue()).decode("utf-8")

                    # Pack all images into a single response
                    output = {'method': method, 'layer': layer,
                              'comp': comp64, 'grey': grey64, 'heatmap': heat64}

    except Exception as ex:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print("exception ", str(exc_tb))
        traceback.print_exception(exc_type, exc_value, exc_tb)
        output = {'error': str(ex)}

    return Response(json.dumps(output), mimetype="application/json")
