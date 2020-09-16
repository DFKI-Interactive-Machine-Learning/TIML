import pytest

import PIL.Image
from PIL.Image import Image

from keras.layers import Dense, InputLayer

from ..classifier import Classifier

import pkg_resources
from typing import Tuple


@pytest.fixture
def sample_image() -> Image:
    image_path = pkg_resources.resource_filename("timl.data", "sample_images/ISIC_0000000.jpeg")
    img = PIL.Image.open(image_path)

    return img


@pytest.fixture(params=[(227, 227), (450, 450)])
def image_size(request) -> Tuple[int, int]:
    return request.param


@pytest.fixture(params=[2, 8])
def n_classes(request) -> int:
    return request.param


@pytest.fixture(params=["VGG16", "RESNET50"])
def classifier(request, image_size, n_classes) -> Classifier:
    from timl.classification.classifier_factory import make_classifier
    out = make_classifier(train_method=request.param, image_size=image_size[0], n_classes=n_classes)
    return out


def test_classifier(classifier: Classifier, sample_image, image_size, n_classes):

    model = classifier.get_model()
    assert model is not None
    assert len(model.layers) > 2

    # The input layer must match the image size
    in_layer: InputLayer = model.layers[0]
    assert type(in_layer) == InputLayer
    assert in_layer.input_shape == (None, image_size[0], image_size[1], 3)
    assert in_layer.output_shape == (None, image_size[0], image_size[1], 3)
    assert not in_layer.trainable

    # The output layer must be a softmax with the correct number of classes.
    out_layer: Dense = model.layers[-1]
    assert type(out_layer) == Dense
    assert out_layer.units == n_classes
    assert out_layer.output_shape == (None, n_classes)
    assert out_layer.trainable

    assert classifier.get_output_size() == n_classes

    # load an image an predict
    img = PIL.Image.open(fp=sample_image)
    prediction, confidence = classifier.predict_image(img)

    for p in prediction:
        assert 0 <= p <= 1.0

    for c in confidence:
        assert -1 <= c <= 1

    # TODO -- generate an heatmap and check resulting size, mode, and pixel values
