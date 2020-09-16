from keras.optimizers import SGD, Nadam, Adadelta, RMSprop
from keras.models import Model
from keras.optimizers import Optimizer

from timl.classification.classifier import Classifier
from timl.classification.vgg16.vgg import VGG16flat

from typing import Tuple, Optional, List


class VGG16FlatClassifier(Classifier):

    def __init__(self, input_dim: Tuple[int, int], n_classes: int):
        super().__init__()

        self._model = VGG16flat(dim=input_dim, fc_shape=[2048, 2048], n_classes=n_classes).model
        # self._model.summary()

        sgd = SGD(lr=1e-5, decay=1e-4, momentum=0.9, nesterov=True)

        self._model.compile(loss='categorical_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])


class VGG16ClassifierML(Classifier):
    """
    Setup a Classifier for multi-label prediction (uses sigmoid at last layer).
    Loss function is binary crossentropy.
    See: https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede
    """

    def is_multilabel(self) -> bool:
        return True

    from keras.callbacks import Callback

    def extra_callbacks(self) -> Optional[List[Callback]]:
        """Sub-classes can optionally return a list o extra Callbacks to be used during training."""
        from keras.callbacks import ReduceLROnPlateau, EarlyStopping

        # Reducing Learning Rate on Plateaus
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_mean_f1_score', mode="max", factor=0.2, patience=5, verbose=1)
        early_stop = EarlyStopping(monitor='val_mean_f1_score', mode="max", patience=10)

        return [reduce_lr, early_stop]

    def quality_metric(self) -> Tuple[str, str]:
        """Subclasses can specify a different metric to use evaluate the 'best' model."""
        return "mean_f1_score", "max"

    def __init__(self,
                 input_dim: Tuple[int, int],
                 n_classes: int,
                 optimizer: Optimizer,
                 weights='imagenet',
                 fc_shape: List[int] = [2048, 2048],
                 fc_initializer: List[str] = ['glorot_uniform', 'glorot_uniform', 'glorot_uniform'],
                 dropouts: List[float] = [0.5, 0.5]):
        super().__init__()

        from timl.classification.metrics import mean_f1_score, loss_1_minus_f1, loss_1mf1_by_bce

        # from keras.applications import VGG16
        # from keras import layers, models

        # # Get VGG16 Keras application
        # base_model = VGG16(weights=weights, input_shape=(input_dim[0], input_dim[1], 3), include_top=False)
        #
        # # Append new layers
        # x = base_model.layers[-1].output
        # x = layers.Flatten(name='flatten')(x)
        # x = layers.Dense(fc_shape[0], activation='relu', name='fc1', kernel_initializer=fc_initializer[0])(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(fc_shape[1], activation='relu', name='fc2', kernel_initializer=fc_initializer[1])(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(n_classes, activation='sigmoid', name='predictions', kernel_initializer=fc_initializer[2])(x)
        #
        # # Instantiate a new model using the defined layers
        # self._model = models.Model(base_model.layers[0].output, x, name=self.__class__.__name__)

        self._model = build_vgg16(n_classes=n_classes,
                                  dim=input_dim,
                                  fc_shape=fc_shape,
                                  dropouts=dropouts,
                                  fc_initializers=fc_initializer,
                                  weights=weights,
                                  final_activation='sigmoid'
                                  )

        # Compile for multi-label entropy
        # Binary cross-entropy as loss
        # E.g.: https://stats.stackexchange.com/questions/357541/what-is-the-difference-between-binary-cross-entropy-and-categorical-cross-entrop
        self._model.compile(#loss='binary_crossentropy',
                            #loss=loss_1_minus_f1,
                            loss=loss_1mf1_by_bce,
                            optimizer=optimizer,
                            metrics=['accuracy', 'binary_accuracy', mean_f1_score])
        print("Compiled.")


def build_vgg16(n_classes: int,
                   dim: Tuple[int, int] = (227, 227),
                   fc_shape: List[int] = [2048, 2048],
                   dropouts: List[float] = [0.5, 0.5],
                   fc_initializers: List[str] = ['glorot_uniform', 'glorot_uniform', 'glorot_uniform'],
                   weights: str = 'imagenet',
                   final_activation: str = 'softmax'
                ) -> Model:
    """
    The factory method to setup a model based on VGG16.

    :param n_classes: The number of predictions for the final softmax
    :param dim: width,height of the input image
    :param fc_shape: the sizes of the last fully connected layers
    :param dropouts: the dropout rate of the last fully connected layers. Same number as the fc_shape.
    :param fc_initializers: the initializer policy for the fully connected plus the final softmax layers. Size must be as the len(fc_shape) + 1
    :param weights: the initial weights of the resnet model (Forwarded to Keras application. Defaults to 'imagenet')
    :param final_activation: The function for the final activation layer. Normally either 'softmax' or 'sigmoid'.
    :return: The built model (not compiled, optimizer not set)
    """

    from keras.applications import VGG16
    from keras import layers, models

    n_dense_layers = len(fc_shape)
    if len(dropouts) != n_dense_layers:
        raise Exception("Wrong number of dropouts. Expected {}, found {}.".format(n_dense_layers, len(dropouts)))

    if len(fc_initializers) != n_dense_layers + 1:
        raise Exception("Wrong number of fc_initializers. Expected {}, found {}".format(n_dense_layers, len(fc_initializers)))

    # Basic model, from which we will extract the layers
    base_model = VGG16(weights=weights, input_shape=(dim[0], dim[1], 3), include_top=False)

    # Append layers to the basic model
    x = base_model.layers[-1].output
    x = layers.Flatten(name='flatten')(x)

    for n in range(n_dense_layers):
        x = layers.Dense(fc_shape[n], activation='relu', name='fc'+str(n), kernel_initializer=fc_initializers[n])(x)
        x = layers.Dropout(dropouts[n])(x)

    x = layers.Dense(n_classes, activation=final_activation, name='predictions', kernel_initializer=fc_initializers[-1])(x)

    # Compose model
    model = models.Model(base_model.layers[0].output, x, name='vgg16')

    return model
