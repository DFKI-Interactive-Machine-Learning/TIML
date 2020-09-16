import pytest
import os

from keras.models import load_model

from timl.classification.classifier import Classifier

#
# Data to use
#SKINCARE_DATA_DIR = get_skincare_datadir()
#MODEL_FILE = os.path.join(SKINCARE_DATA_DIR,
#                          "ISIC_Challenge_2019/Models/Model-ISIC2019-8cls-VGG16flat-227px-20k/0-keras_model.h5")

SAMPLE_IMAGE_PATH = "sample_images/ISIC_0000000.jpeg"


def test_xai_methods(tmp_path):
    import PIL.Image

    # Load trained model
    model = load_model(MODEL_FILE)

    # Initialize model
    classifier = Classifier()
    classifier._model = model

    # Retrieve the model input dimensions
    w, h, _ = classifier.get_input_size()

    #
    # Load sample image
    image = PIL.Image.open(SAMPLE_IMAGE_PATH)

    print("Generating XAI images in {}".format(tmp_path))

    for xai_method, extra_params in [
        ('gradcam', {'layer_name': 'block5_conv3'}),
        ('rise', {'N': 20})
    ]:

        print("Generating XAI images for method {} with parameters {}".format(xai_method, extra_params))

        grey_map, heat_map, composite = classifier.generate_heatmap(image=image, method=xai_method, **extra_params)

        assert grey_map.mode == 'L'
        assert heat_map.mode == 'RGB'
        assert composite.mode == 'RGB'

        assert grey_map.size == (w, h)
        assert heat_map.size == (w, h)
        assert composite.size == (w, h)

        grey_map.save(os.path.join(tmp_path, "{}-greymap.png".format(xai_method)))
        heat_map.save(os.path.join(tmp_path, "{}-heatmap.png".format(xai_method)))
        composite.save(os.path.join(tmp_path, "{}-composite.png".format(xai_method)))
