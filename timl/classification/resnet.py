from keras.models import Model
from keras.optimizers import Optimizer

from timl.classification.classifier import Classifier

from typing import Tuple


def build_resnet50(n_classes: int,
                   dim: Tuple[int, int] = (227, 227),
                   fc_shape: Tuple[int, int] = (2048, 2048),
                   dropouts: Tuple[float, float] = (0.5, 0.5),
                   weights: str = 'imagenet',
                   fc_initializers: Tuple[str, str, str] = ('glorot_uniform', 'glorot_uniform', 'glorot_uniform'),
                   final_activation: str = 'softmax'
                   ) -> Model:
    """
    The factory method to setup a model based on RESNET50.

    :param n_classes: The number of predictions for the final softmax
    :param dim: width,height of the input image
    :param fc_shape: the sizes of the last fuly connected layers
    :param dropouts: the dropout rate of the last fully connected layers
    :param weights: the initial weights of the resnet model (defaults to imagenet)
    :param fc_initializers: the initializer policy for the fully connected plus the final softmax layers
    :param final_activation: The function for the final activation layer. Normally either softmax or sigmoid.
    :return: The built model (not compiled, optimizer not set)
    """

    from keras.applications import ResNet50
    from keras import layers, models

    # Basic model, from which we will extract the layers
    base_model = ResNet50(weights=weights, input_shape=(dim[0], dim[1], 3), include_top=False)

    # Append layers to the basic model
    x = base_model.layers[-1].output
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(fc_shape[0], activation='relu', name='fc1', kernel_initializer=fc_initializers[0])(x)
    x = layers.Dropout(dropouts[0])(x)
    x = layers.Dense(fc_shape[1], activation='relu', name='fc2', kernel_initializer=fc_initializers[1])(x)
    x = layers.Dropout(dropouts[0])(x)
    x = layers.Dense(n_classes, activation=final_activation, name='predictions', kernel_initializer=fc_initializers[2])(x)

    # Compose model
    model = models.Model(base_model.layers[0].output, x, name='resnet50')

    return model


class ResNet50Classifier(Classifier):

    def __init__(self,
                 input_dim: Tuple[int, int],
                 n_classes: int,
                 optimizer: Optimizer,
                 loss_function: str = "categorical_crossentropy",
                 fc_shape: Tuple[int, int] = [2048, 2048],
                 activation: str = 'softmax',
                 metric_list: list = ['accuracy'],
                 ):
        super().__init__()

        self._model = build_resnet50(n_classes=n_classes, dim=input_dim, fc_shape=fc_shape, final_activation=activation)

        self._model.compile(loss=loss_function,
                            optimizer=optimizer,
                            metrics=metric_list)


class ResNet50ClassifierML(Classifier):

    def is_multilabel(self) -> bool:
        return True

    def quality_metric(self) -> Tuple[str, str]:
        return "mean_f1_score", "max"

    def __init__(self,
                 input_dim: Tuple[int, int],
                 n_classses: int,
                 optimizer: Optimizer,
                 loss_function: str = "binary_crossentropy",
                 fc_shape: Tuple[int, int] = [2048, 2048],
                 activation: str = 'sigmoid',
                 metric_list: list = ['accuracy'],
                 ):
        super().__init__()

        self._model = build_resnet50(n_classes=n_classses, dim=input_dim, fc_shape=fc_shape, final_activation=activation)

        self._model.compile(loss=loss_function,
                            optimizer=optimizer,
                            metrics=metric_list)

