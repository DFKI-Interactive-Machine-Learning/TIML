from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.models import Model
from keras.applications import VGG16
from keras.backend import int_shape
from keras import layers, models, Input, Model


class SkinLesionModel:
    def __init__(self, dim=(227, 227)):
        self.inputs = self.get_model_input(dim)

    def get_model_input(self, dim):
        inputs = Input(shape=dim+(3,))
        return inputs

    def append_output2model(self, base_model, n_out_classes=2):
        # for the simplest case of two classes
        pred_malign = Dense(n_out_classes, activation='softmax', name='output')(base_model)
        model = Model(inputs=self.inputs, outputs=pred_malign)
        return model


class Vgg16(SkinLesionModel):
    def __init__(self, n_classes: int, dim=(227, 227), fc_shape=[2048, 2048], weights='imagenet',
                 fc_initializer=['glorot_uniform', 'glorot_uniform']):
        super().__init__(dim)
        self.fc_shape = fc_shape
        self.weights = weights
        self.fc_initializer = fc_initializer
        self.base_model = self.get_base_model()

        self.model = self.append_output2model(self.base_model, n_out_classes=n_classes)

    def get_base_model(self):
        # Pretrained Convolutional Block
        base_model = VGG16(weights=self.weights, input_shape=int_shape(self.inputs)[1:], include_top=False)
        # Top Fully Connected Block with out classifier
        conv_out = base_model(self.inputs)
        base_model = Flatten()(conv_out)

        base_model = Dense(self.fc_shape[0],
                           activation='relu',
                           name='fc1',
                           kernel_initializer=self.fc_initializer[0])(base_model)
        base_model = Dropout(0.5)(base_model)
        base_model = Dense(self.fc_shape[1],
                           activation='relu',
                           name='fc2',
                           kernel_initializer=self.fc_initializer[1])(base_model)
        base_model = Dropout(0.5)(base_model)
        return base_model


class VGG16flat:
    def __init__(self, n_classes: int,
                 dim=(227, 227),
                 fc_shape=[2048, 2048],
                 weights='imagenet',
                 fc_initializer=['glorot_uniform', 'glorot_uniform', 'glorot_uniform']
                 ):

        # Get VGG16 Keras application
        base_model = VGG16(weights=weights, input_shape=(dim[0], dim[1], 3), include_top=False)

        # Append layers
        x = base_model.layers[-1].output
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(fc_shape[0], activation='relu', name='fc1', kernel_initializer=fc_initializer[0])(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(fc_shape[1], activation='relu', name='fc2', kernel_initializer=fc_initializer[1])(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(n_classes, activation='softmax', name='predictions', kernel_initializer=fc_initializer[2])(x)

        # Compose model
        self.model = models.Model(base_model.layers[0].output, x, name='vgg16flat')


def build_vgg16md_model(n_classes: int, dim=(227, 227), fc_shape=[2048, 2048], weights='imagenet',
                      fc_initializer=['glorot_uniform', 'glorot_uniform'], meta_len=None):
    """Function to build a VGG16 model which takes as a second input a vector to concatenate
    with the first fully connected layer after the convolution stage."""

    input_layer = Input(shape=dim + (3,))

    # .....shape of meta input
    input_meta = Input((meta_len,))# concatenate with flattened input from meta data
    # base_model = concatenate([base_model, input_meta], axis=-1)

    # Pretrained Convolutional Block
    base_model = VGG16(weights=weights, input_shape=int_shape(input_layer)[1:], include_top=False)

    # Top Fully Connected Block with out classifier
    conv_out = base_model(input_layer)
    base_model = Flatten()(conv_out)

    base_model = Dense(fc_shape[0],
                       activation='relu',
                       name='fc1',
                       kernel_initializer=fc_initializer[0])(base_model)

    # concatenate with Fc1 input from meta data
    base_model = concatenate([base_model, input_meta], axis=-1)

    base_model = Dense(fc_shape[0],
                       activation='relu',
                       name='fcc',
                       kernel_initializer=fc_initializer[0])(base_model)

    base_model = Dropout(0.5)(base_model)
    base_model = Dense(fc_shape[1],
                       activation='relu',
                       name='fc2',
                       kernel_initializer=fc_initializer[1])(base_model)
    base_model = Dropout(0.5)(base_model)

    out_softmax = Dense(n_classes, activation='softmax', name='output')(base_model)

    meta_model = Model(inputs=[input_layer, input_meta], outputs=[out_softmax])

    return meta_model
