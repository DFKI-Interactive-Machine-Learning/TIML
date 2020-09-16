import keras

from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3

from keras.utils import plot_model

models = {
    "VGG16": [VGG16, {'include_top': True}],
    "VGG16notop": [VGG16, {'include_top': False}],
    "VGG19": [VGG19, {'include_top': True}],
    "VGG19notop": [VGG19, {'include_top': False}],
    "ResNet50": [ResNet50, {'include_top': True}],
    "ResNet50notop": [ResNet50, {'include_top': False}],
    "InceptionV3": [InceptionV3, {'include_top': True}],
    "InceptionV3notop": [InceptionV3, {'include_top': False}],
}

for m_name in models.keys():
    print("Generating plot for model " + m_name)
    m_class, params = models[m_name]
    m_instance = m_class(weights=None, **params)  # type: keras.models.Model
    print(m_instance.summary())

    plot_filename = "KerasModel-" + m_name + ".png"
    plot_model(model=m_instance, to_file=plot_filename, show_shapes=True, show_layer_names=True)



