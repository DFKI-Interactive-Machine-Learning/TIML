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


BINARY_CLASSIFICATION_MODEL_PATH = timl.networking.config.get_server_classification_model_path()
BINARY_CLASSES = ["BEN", "MAL"]
BINARY_CLASSES_DESCR = ["benign", "malignant"]

ISIC2019_CLASSIFICATION_MODEL_PATH = timl.networking.config.get_server_isic_2019_classification_model_path()
ISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
ISIC2019_CLASSES_DESCR = [
    "Melanoma",
    "Melanocytic nevus",
    "Basal cell carcinoma",
    "Actinic keratosis",
    "Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)",
    "Dermatofibroma",
    "Vascular lesion",
    "Squamous cell carcinoma"
]

SEGMENTATION_WEIGHTS_PATH = timl.networking.config.get_segmentation_weights_path()

STATIC_HTML_DIR = timl.networking.config.get_server_static_dir()
STATIC_HTML_URL = timl.networking.config.get_server_static_url()

#
# All the following are global variables used to store the Keras models and avoid re-loading at every call.
# Loading must be done in the call invocation thread. Can not be done here in the main execution.
# All the model initializations and predictions must be done in a properly set Tensorflow Session and Graph.
# See here: https://github.com/keras-team/keras/issues/8538
# and here: https://github.com/keras-team/keras/issues/2397#issuecomment-385317242

#
# The binary classifier
binary_classification_session = None  # type: Optional[tf.Session]
binary_classification_classifier = None  # type: Optional[Classifier]

#
# The 8-class classifier
isic2019_classification_session = None  # type: Optional[tf.Session]
isic2019_classification_classifier = None  # type: Optional[Classifier]

#
# The UNET segmenter
# Will be initialized at first usage
segmenter = None  # type: Optional[UNetSegmenter]
segmenter_session = None  # type: Optional[tf.Session]

#
# Feature Extraction models and sessions
feature_extractors = {}  # type: Dict[str, keras.Model]
feature_extractors_session = {}  # type: Dict[str, tf.Session]

#
# Check HTML dir presence

html_abs_path = os.path.join(os.getcwd(), STATIC_HTML_DIR)
if not os.path.exists(html_abs_path):
    print("Couldn't find dir '{}' for static html files.".format(html_abs_path))
    exit(10)
print("Static html files will be searched in '{}' and accessible via URL path '/{}'".
      format(html_abs_path, STATIC_HTML_URL))

# The Flask app, used to serve.
app = Flask(__name__)

#import timl.networking.proxy
#
# Routing functions
#


@app.route('/ping')
def hello_world():
    return 'pong'


@app.route('/echo/<myparameter>')
def echo(myparameter):
    return "You told me " + myparameter


@app.route('/model_info')
@app.route('/skin/ModelInfo')
@app.route('/timl/ModelInfo')
def model_info():

    out = {
        "binary_classification_model": BINARY_CLASSIFICATION_MODEL_PATH,
        "binary_classification_classes": BINARY_CLASSES,
        "binary_classification_classes_description": BINARY_CLASSES_DESCR,
        "eight_class_classification_model": ISIC2019_CLASSIFICATION_MODEL_PATH,
        "eight_class_classification_classes": ISIC2019_CLASSES,
        "eight_class_classification_classes_description": ISIC2019_CLASSES_DESCR,
        "segmentation_model": SEGMENTATION_WEIGHTS_PATH,
        "feature_extraction_classes": [f for f in FEATURE_CLASSES]
    }

    return Response(json.dumps(out), mimetype="application/json")


@app.route('/' + STATIC_HTML_URL + '/<path:filename>')
@app.route('/skin/<path:filename>')
def static_html_loader(filename):
    return send_from_directory(directory=html_abs_path, filename=filename)


# this is the pseudo-call triggered by "Explain"
@app.route('/IML/XAI2/Classifier', methods=['POST'])
def skincare_xaiclassify():
    file = request.files['file']
    args = request.args.to_dict()
    argsstring = '&'.join("=".join((str(k),str(v))) for k,v in args.items())
    return post(f'http://dfki.de/IML/XAI2/Classifier?'+argsstring, files={'file': file}).content


'''In the following the additional routing items are only needed 
    if this server runs without a separate HTTP server (XAMPP)'''

# Internal re-routing to contact the ABCD server
@app.route('/abcd', methods=['POST'])
@app.route('/skin/abcd', methods=['POST'])
@app.route('/timl/abcd', methods=['POST'])
def abcd():
    file = request.files['file']
    return post(f'http://localhost:59/abcd', files={'file': file}).content
    #return post(f'http://localhost:5001/abcd', files={'file': file}).content



@app.route('/classify/binary', methods=['POST'])
@app.route('/skin/Classifier', methods=['POST'])
@app.route('/timl/Classifier', methods=['POST'])
def classify_binary():
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
                            binary_classification_model = keras.models.load_model(filepath=BINARY_CLASSIFICATION_MODEL_PATH)
                            binary_classification_classifier = Classifier()
                            binary_classification_classifier._model = binary_classification_model

                assert (binary_classification_classifier is not None) and (binary_classification_session is not None)

                with binary_classification_session.as_default():
                    with binary_classification_session.graph.as_default():
                        predicted_classes, confidences = binary_classification_classifier.predict_image(PIL.Image.open(img_full_tmp_path))

                # output["prediction"] = json.dumps(predicted_classes)
                # output["confidence"] = json.dumps(confidences)
                output["prediction"] = predicted_classes
                output["confidence"] = confidences

        except Exception as ex:
            print(ex)
            output["error"] = str(ex)

    return Response(json.dumps(output), mimetype="application/json")


@app.route('/classify/eight_class', methods=['POST'])
@app.route('/skin/Classifier8', methods=['POST'])
@app.route('/timl/Classifier8', methods=['POST'])
def classify_isic2019():
    global isic2019_classification_session
    global isic2019_classification_classifier

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

                # Initialize the 8-class classifier, if needed
                if isic2019_classification_classifier is None:

                    isic2019_classification_session = tf.Session(graph=tf.Graph())
                    with isic2019_classification_session.as_default():
                        with isic2019_classification_session.graph.as_default():
                            isic2019_classification_model = keras.models.load_model(filepath=ISIC2019_CLASSIFICATION_MODEL_PATH)
                            isic2019_classification_classifier = Classifier()
                            isic2019_classification_classifier._model = isic2019_classification_model

                assert (isic2019_classification_classifier is not None) and (isic2019_classification_session is not None)

                with isic2019_classification_session.as_default():
                    with isic2019_classification_session.graph.as_default():
                        predicted_classes, confidences = isic2019_classification_classifier.predict_image(PIL.Image.open(img_full_tmp_path))

                #output["prediction"] = json.dumps(predicted_classes)
                #output["confidence"] = json.dumps(confidences)
                output["prediction"] = predicted_classes
                output["confidence"] = confidences

        except Exception as ex:
            print(ex)
            output["error"] = str(ex)

    return Response(json.dumps(output), mimetype="application/json")


@app.route('/segment', methods=['POST'])
@app.route('/skin/Segment', methods=['POST'])
@app.route('/timl/Segment', methods=['POST'])
def segment():
    global segmenter
    global segmenter_session

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
                in_image = PIL.Image.open(fp=img_full_tmp_path)

                # Initialize the segmentation model, if needed
                if segmenter is None:
                    segmenter_session = tf.Session(graph=tf.Graph())
                    with segmenter_session.as_default():
                        with segmenter_session.graph.as_default():
                            segmenter = UNetSegmenter(weights_file_path=SEGMENTATION_WEIGHTS_PATH)

                with segmenter_session.as_default():
                    with segmenter_session.graph.as_default():
                        segmented_image = segmenter.segment_image(im=in_image)

                # Save the mask image into a temporary memory buffer and return it as image.
                mask_img_buf = BytesIO()
                segmented_image.save(fp=mask_img_buf, format='PNG')
                mask_img_buf.seek(0)
                return send_file(filename_or_fp=mask_img_buf, mimetype="image/png")

        except Exception as ex:
            print(ex)
            output["error"] = str(ex)

    return Response(json.dumps(output), mimetype='application/json')


@app.route('/extract_feature/<feature_class>', methods=['POST'])
@app.route('/skin/ExtractFeature/<feature_class>', methods=['POST'])
@app.route('/timl/ExtractFeature/<feature_class>', methods=['POST'])
def extract_feature(feature_class):
    global feature_extractors
    global feature_extractors_session

    import PIL.Image
    from timl.features.featureextraction import build_model, extract_feature

    output = {}

    if "file" not in request.files:
        output["error"] = "Missing 'file'"
    elif feature_class not in FEATURE_CLASSES:
        output["error"] = "Feature named '{}' is unknown. Must be one of {}.".format(feature_class, FEATURE_CLASSES)
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
                in_image = PIL.Image.open(fp=img_full_tmp_path)

                # Initialize the feature extraction model, if needed.
                if feature_class not in feature_extractors:
                    tf_session = tf.Session(graph=tf.Graph())
                    feature_extractors_session[feature_class] = tf_session
                    with tf_session.as_default():
                        with tf_session.graph.as_default():
                            feature_extractors[feature_class] = build_model(feature_class=feature_class)

                with feature_extractors_session[feature_class].as_default():
                    with feature_extractors_session[feature_class].graph.as_default():
                        feature_mask_img = extract_feature(img=in_image, model=feature_extractors[feature_class])

                # Save the mask image into a temporary memory buffer and return it as image.
                feature_mask_img_buf = BytesIO()
                feature_mask_img.save(fp=feature_mask_img_buf, format='PNG')
                feature_mask_img_buf.seek(0)
                return send_file(filename_or_fp=feature_mask_img_buf, mimetype="image/png")

        except Exception as ex:
            output["error"] = str(ex)

    return Response(json.dumps(output), mimetype='application/json')


@app.route('/explain', methods=['POST'])
@app.route('/skin/explain', methods=['POST'])
@app.route('/timl/explain', methods=['POST'])
def explain():
    data = request.args.to_dict()
    print("explpain - data ", data)

    output = {}

    if "file" not in request.files:
        output["error"] = "Missing 'file'"
        return Response(json.dumps(output), mimetype="application/json")
    else:
        file = request.files['file']

        if 'visual' in data and data['visual'] == 'gradcam':
            if 'layers' in data:
                return explain_gradcam_layers(file, data)
            else:
                return explain_gradcam(file, data)
        else:
            return explain_rise(file, data)


def explain_gradcam_layers(file, data):
    global isic2019_classification_session
    global isic2019_classification_classifier

    from timl.classification.classifier import Classifier

    output = {}

    filename = secure_filename(file.filename)
    output["filename"] = filename

    #data = request.args.to_dict()
    xai_method = 'gradcam'
    layer_names = ['block5_pool', 'block5_conv3', 'block5_conv2', 'block5_conv1']
    layers = 4

    try:
        if 'layers' in data:
            layers = int(data['layers'])

        print("explain image classification, method={}, layers={}, model={}".format(xai_method, layers, ISIC2019_CLASSIFICATION_MODEL_PATH))

        # Save the image into a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('Created temporary directory', tmpdirname)

            img_full_tmp_path = os.path.join(tmpdirname, filename)

            file.save(img_full_tmp_path)

            # Initialize the 8-class classifier, if needed
            if isic2019_classification_classifier is None:
                isic2019_classification_session = tf.Session(graph=tf.Graph())
                with isic2019_classification_session.as_default():
                    with isic2019_classification_session.graph.as_default():
                        isic2019_classification_model = keras.models.load_model(
                            filepath=ISIC2019_CLASSIFICATION_MODEL_PATH)
                        isic2019_classification_classifier = Classifier()
                        isic2019_classification_classifier._model = isic2019_classification_model

            assert (isic2019_classification_classifier is not None) and (
                        isic2019_classification_session is not None)

            # Retrieve the model input dimensions
            w, h, _ = isic2019_classification_classifier.get_input_size()
            layer_maps = []

            for layer_name in layer_names[0:layers]:
                extra_params = {'layer_name': layer_name}

                with isic2019_classification_session.as_default():
                    with isic2019_classification_session.graph.as_default():
                        grey_map, heat_map, composite = \
                            isic2019_classification_classifier.generate_heatmap(
                                image=PIL.Image.open(img_full_tmp_path),
                                method=xai_method,
                                **extra_params)

                        assert grey_map.mode == 'L'
                        assert heat_map.mode == 'RGB'
                        assert composite.mode == 'RGB'

                        assert grey_map.size == (w, h)
                        assert heat_map.size == (w, h)
                        assert composite.size == (w, h)

                        comp_buff = BytesIO()
                        composite.save(fp=comp_buff, format='PNG')
                        comp_buff.seek(0)
                        comp64 = base64.b64encode(comp_buff.getvalue()).decode("utf-8")

                        # grey_buff = BytesIO()
                        # grey_map.save(fp=grey_buff, format='PNG')
                        # grey_buff.seek(0)
                        # grey64 = base64.b64encode(grey_buff.getvalue()).decode("utf-8")
                        #
                        # heat_buff = BytesIO()
                        # heat_map.save(fp=heat_buff, format='PNG')
                        # heat_buff.seek(0)
                        # heat64 = base64.b64encode(heat_buff.getvalue()).decode("utf-8")

                        layer_maps.append({'layer_name': layer_name, 'composite': comp64})

                output = {'method': xai_method, 'layers': layer_maps}
            return Response(json.dumps(output), mimetype="application/json")

    except Exception as ex:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print("exception ", str(exc_tb))
        traceback.print_exception(exc_type, exc_value, exc_tb)
        output = {'error': str(ex)}

    return Response(json.dumps(output), mimetype="application/json")


def explain_gradcam(file, data):
    global isic2019_classification_session
    global isic2019_classification_classifier

    from timl.classification.classifier import Classifier

    output = {}

    filename = secure_filename(file.filename)
    output["filename"] = filename

    #data = request.args.to_dict()
    xai_method = 'gradcam'

    if 'args' in data:
        extra_params = json.loads(data['args'])
    else:
        extra_params = {'layer_name': 'block5_conv3'}

    try:

        print("explain image classification, method={}, args={}, model={}".format(xai_method, extra_params, ISIC2019_CLASSIFICATION_MODEL_PATH))

        # Save the image into a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('Created temporary directory', tmpdirname)

            img_full_tmp_path = os.path.join(tmpdirname, filename)

            file.save(img_full_tmp_path)

            # Initialize the 8-class classifier, if needed
            if isic2019_classification_classifier is None:
                isic2019_classification_session = tf.Session(graph=tf.Graph())
                with isic2019_classification_session.as_default():
                    with isic2019_classification_session.graph.as_default():
                        isic2019_classification_model = keras.models.load_model(
                            filepath=ISIC2019_CLASSIFICATION_MODEL_PATH)
                        isic2019_classification_classifier = Classifier()
                        isic2019_classification_classifier._model = isic2019_classification_model

            assert (isic2019_classification_classifier is not None) and (
                        isic2019_classification_session is not None)

            # Retrieve the model input dimensions
            w, h, _ = isic2019_classification_classifier.get_input_size()

            with isic2019_classification_session.as_default():
                with isic2019_classification_session.graph.as_default():
                    grey_map, heat_map, composite = \
                        isic2019_classification_classifier.generate_heatmap(
                            image=PIL.Image.open(img_full_tmp_path),
                            method=xai_method,
                            **extra_params)

                    assert grey_map.mode == 'L'
                    assert heat_map.mode == 'RGB'
                    assert composite.mode == 'RGB'

                    assert grey_map.size == (w, h)
                    assert heat_map.size == (w, h)
                    assert composite.size == (w, h)

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

                    output = {'method': xai_method, 'args': extra_params, 'comp': comp64, 'grey': grey64, 'heatmap': heat64}
                    return Response(json.dumps(output), mimetype="application/json")

    except Exception as ex:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print("exception ", str(exc_tb))
        traceback.print_exception(exc_type, exc_value, exc_tb)
        output = {'error': str(ex)}

    return Response(json.dumps(output), mimetype="application/json")


def explain_rise(file, data):
    global binary_classification_session
    global binary_classification_classifier

    from timl.classification.classifier import Classifier

    output = {}

    filename = secure_filename(file.filename)
    output["filename"] = filename

    #data = request.args.to_dict()
    xai_method = 'rise'

    try:
        if 'args' in data:
            extra_params = json.loads(data['args'])
        else:
            extra_params = {'N': 10}

        print("explain image classification, method={}, args={}, model={}".format(xai_method, extra_params, BINARY_CLASSIFICATION_MODEL_PATH))

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
                        binary_classification_model = keras.models.load_model(
                            filepath=BINARY_CLASSIFICATION_MODEL_PATH)
                        binary_classification_classifier = Classifier()
                        binary_classification_classifier._model = binary_classification_model

            assert (binary_classification_classifier is not None) and (binary_classification_session is not None)

            # Retrieve the model input dimensions
            w, h, _ = binary_classification_classifier.get_input_size()
            layer_maps = []

            with binary_classification_session.as_default():
                with binary_classification_session.graph.as_default():
                    grey_map, heat_map, composite = \
                        binary_classification_classifier.generate_heatmap(
                            image=PIL.Image.open(img_full_tmp_path),
                            method='rise',
                            **extra_params)

                    assert grey_map.mode == 'L'
                    assert heat_map.mode == 'RGB'
                    assert composite.mode == 'RGB'

                    assert grey_map.size == (w, h)
                    assert heat_map.size == (w, h)
                    assert composite.size == (w, h)

                    comp_buff = BytesIO()
                    composite.save(fp=comp_buff, format='PNG')
                    comp_buff.seek(0)
                    comp64 = base64.b64encode(comp_buff.getvalue()).decode("utf-8")
                    #
                    # grey_buff = BytesIO()
                    # grey_map.save(fp=grey_buff, format='PNG')
                    # grey_buff.seek(0)
                    # grey64 = base64.b64encode(grey_buff.getvalue()).decode("utf-8")
                    #
                    # heat_buff = BytesIO()
                    # heat_map.save(fp=heat_buff, format='PNG')
                    # heat_buff.seek(0)
                    # heat64 = base64.b64encode(heat_buff.getvalue()).decode("utf-8")

                    layer_maps.append({'layer_name': 'rise heatmap', 'composite': comp64})

            output = {'method': xai_method, 'args': extra_params, 'layers': layer_maps}

            return Response(json.dumps(output), mimetype="application/json")

    except Exception as ex:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print("exception ", str(exc_tb))
        traceback.print_exception(exc_type, exc_value, exc_tb)
        output = {'error': str(ex)}

    return Response(json.dumps(output), mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='88')