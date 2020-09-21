import pytest
import os

from PIL.Image import Image

from timl.classification.classifier import Classifier

# Fixtures from other tests
from timl.common.test.test_augmentation import sample_image


def test_xai_methods(tmp_path, sample_image: Image):

    from timl.classification.classifier_factory import make_classifier
    classifier = make_classifier(train_method="VGG16", image_size=227, n_classes=8)

    print("Generating XAI images in {}".format(tmp_path))

    # Retrieve the model input dimensions
    w, h, _ = classifier.get_input_size()

    for xai_method, extra_params in [
        ('gradcam', {'layer_name': 'block5_conv3'}),
        ('rise', {'N': 20})
    ]:

        print("Generating XAI images for method {} with parameters {}".format(xai_method, extra_params))

        grey_map, heat_map, composite = classifier.generate_heatmap(image=sample_image,
                                                                    method=xai_method,
                                                                    **extra_params)

        assert grey_map.mode == 'L'
        assert heat_map.mode == 'RGB'
        assert composite.mode == 'RGB'

        assert grey_map.size == (w, h)
        assert heat_map.size == (w, h)
        assert composite.size == (w, h)

        grey_map.save(os.path.join(tmp_path, "{}-greymap.png".format(xai_method)))
        heat_map.save(os.path.join(tmp_path, "{}-heatmap.png".format(xai_method)))
        composite.save(os.path.join(tmp_path, "{}-composite.png".format(xai_method)))
