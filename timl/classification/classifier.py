from abc import ABC

import os
import re

from typing import Optional
from typing import List
from typing import Tuple

import numpy as np

import keras.models
from keras.engine.training import Model
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

from PIL.Image import Image
import PIL
from timl.common.datageneration import SkincareDataGenerator


#
#
#

class TrainHistory(Callback):
    """Holds the sequence of losses and accuracies for both training (every batch) and validation (each epoch)"""

    def __init__(self):
        super().__init__()

        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_batch_end(self, batch, logs=None):
        # on_batch_end: logs include loss, and optionally acc (if accuracy monitoring is enabled).
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs=None):
        # on_epoch_end: logs include acc and loss, and optionally include val_loss (if validation is enabled in fit),
        # and val_acc (if validation and accuracy monitoring are enabled)
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc'))


class TrainResults:
    """Holds the results of a training session."""

    def __init__(self, last_model: Model, best_model: Model, best_epoch: int, train_history: TrainHistory):
        self.last_model = last_model
        self.best_model = best_model
        self.best_epoch = best_epoch
        self.iteration_history = train_history

    def save_best_model(self, out_dir: str, prefix: Optional[str] = None, save_weights: bool = False) -> Tuple[str, Optional[str]]:
        """
        Saves keras_model.h5 and keras_weigth.h5 to the specified dir, using the given prefix.

        :param out_dir:
        :param prefix:
        :return: The filenames of the model and of the weigths files, as string 2-tuple
        """

        if prefix is None:
            prefix = ""

        model_save_filename = prefix + "keras_model.h5"
        self.best_model.save(os.path.join(out_dir, model_save_filename))

        if save_weights:
            weights_save_filename = prefix + "keras_weights.h5"
            self.best_model.save_weights(os.path.join(out_dir, weights_save_filename))
        else:
            weights_save_filename = None

        return model_save_filename, weights_save_filename


class Classifier(ABC):
    """This is the abstract class representing a Classifier.
    Subclassers must "only" set the field self._model in the constructor.
    """

    def __init__(self):
        self._model = None  # type: Optional[Model]
        # Will be set and used only if the GradCAM XAI method is used
        self._gracam_modified_model = None  # type: Optional[Model]

    def load_model(self, model_filepath: str) -> Model:
        from timl.classification.metrics import mean_f1_score, loss_1_minus_f1, loss_1mf1_by_bce

        self._model = keras.models.load_model(filepath=model_filepath,
                                              custom_objects={"mean_f1_score": mean_f1_score,
                                                              "loss_1_minus_f1": loss_1_minus_f1,
                                                              "loss_1mf1_by_bce": loss_1mf1_by_bce})
        return self._model

    def is_multilabel(self) -> bool:
        """
        A classifier can be configured to perform one-class classification (softmax) or multi-label classification (sigmoid).
        By default, for backward compatibility, classification is on single class (1-hot).
        Subclasses must override is configured for multi-label
        :return: True if the classifier is configured for multi-label
        """
        return False

    def extra_callbacks(self) -> Optional[List[Callback]]:
        """Sub-classes can optionally return a list o extra Callbacks to be used during training."""
        return None

    def quality_metric(self) -> Tuple[str, str]:
        """Subclasses can specify a different metric (and target mode: max, min, auto) to use evaluate the 'best' model.
        By default, returns ('acc', 'max')"""
        return "acc", "max"

    def _init_gradcam_model(self):
        """This method instantiate a 'dummy' model, by modifiying the original one,
        in order to correctly compute the GradCAM saliency maps.
        This is invoked only the first time a GradCAM image is needed."""

        from vis.utils import utils

        # How to clone a model:
        # https://stackoverflow.com/questions/54366935/make-a-deep-copy-of-a-keras-model-in-python

        self._gracam_modified_model = keras.models.clone_model(self._model)  # type: keras.Model
        self._gracam_modified_model.layers[-1].activation = keras.activations.linear
        # Do not need to compile
        # self._gracam_modified_model.compile(...)
        self._gracam_modified_model = utils.apply_modifications(self._gracam_modified_model)
        # Transfer weights
        self._gracam_modified_model.set_weights(self._model.get_weights())

    def get_model(self) -> Optional[Model]:
        return self._model

    def train(self, out_dir: str, train_generator: SkincareDataGenerator, val_generator: SkincareDataGenerator,
              epochs: int,
              n_cpus: int,
              class_freq_distr: List[float]) -> TrainResults:
        """
        :param out_dir: The directory that will be filled out with the training results (model, weights, tensor borads DB, ...)
        :param train_generator:
        :param val_generator:
        :param epochs:
        :param n_cpus: The number of CPUs to use in parallel for keras.fit()
        :param class_freq_distr: The normalized frequency distribution of the input classes. Must be a vector of elements in [0.0,1.0], summing to 1.0.
        :return:
        """

        if not os.path.exists(out_dir):
            raise Exception("Directory '{}' doesn't exist".format(out_dir))

        if not os.path.isdir(out_dir):
            raise Exception("Path '{}' is not a directory".format(out_dir))

        if self._model is None:
            raise Exception("The _model field is still None. Please, set it in the constructor.")

        #
        # Compute the class weights, setting the weight for the class with the highest number of samples to 1.0
        class_freq_max = max(class_freq_distr)
        class_weights = [class_freq_max / p if p != 0 else 0 for p in class_freq_distr]

        print("Normalized weights:")
        print(class_weights)
        # Convert into a dictionary
        class_weights_dict = {i: p for i, p in enumerate(class_weights)}
        print("Weights dict:")
        print(class_weights_dict)

        #
        # Prepare a temporary directory where we will store all our intermediate models.
        tmp_path = os.path.join(out_dir, "tmp_models")
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        print("Saving intermediate models to: {}".format(tmp_path))

        metric, metric_mode = self.quality_metric()
        print("Quality metric is {}".format(metric))

        # See: https://keras.io/callbacks/#modelcheckpoint
        best_model_template_filename = os.path.join(tmp_path, "best_model-{epoch:04d}.hdf5")
        checkpoint_callback = ModelCheckpoint(best_model_template_filename,
                                              monitor='val_'+metric, mode=metric_mode,
                                              verbose=1, save_best_only=True)

        #
        # This instance will collect the evolution of loss and accuracies
        train_itr_history = TrainHistory()

        #
        # This callback with create information for the Tensor Board.
        # See: https://www.tensorflow.org/guide/summaries_and_tensorboard
        tensorboard_path = os.path.join(out_dir, "tensorboard")
        if not os.path.exists(tensorboard_path):
            os.mkdir(tensorboard_path)
        tensor_board_cb = TensorBoard(log_dir=tensorboard_path, write_images=True, update_freq='epoch')
        # update_freq: 'batch' or 'epoch' or integer

        #
        # Callbacks
        callbacks = [checkpoint_callback, train_itr_history, tensor_board_cb]

        extra_callbacks = self.extra_callbacks()
        if extra_callbacks is not None:
            print("Adding {} callbacks...".format(len(extra_callbacks)))
            for i, cb in enumerate(extra_callbacks):
                print(i, cb.__class__.__name__)
            callbacks = callbacks + extra_callbacks

        #
        # Do the real thing
        self._model.fit_generator(generator=train_generator,
                                  validation_data=val_generator,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=1,
                                  class_weight=class_weights_dict,
                                  workers=n_cpus, use_multiprocessing=True)

        # DIIIIIIRTY HACK! The ModelCheckpoint might be still writing the file. So we wait a bit...
        import time
        time.sleep(2.0)

        #
        # Get the list of saved models, and get the last (alphabetically) as the best
        model_files = os.listdir(path=tmp_path)
        print("All models: ", model_files)

        if len(model_files) < 1:
            raise Exception("No Models have been saved in {}! There should be at least one.".format(tmp_path))

        best_model_filename = sorted(model_files)[-1]  # the most recent is the last in the list
        best_model_path = os.path.join(tmp_path, best_model_filename)

        # Extract epoch number from the filename
        match_expression = "best_model-(\\d+).hdf5"
        res = re.match(match_expression, best_model_filename)
        if res is None:
            raise Exception("Can not extract epoch number from recovered filename {}. Expected a pattern like {}".format(best_model_filename, match_expression))

        best_epoch = int(res.group(1))

        #
        # Load back the best model
        print("Loading back the best from {}".format(best_model_path))
        last_model = self._model

        best_model = self.load_model(model_filepath=best_model_path)

        #
        # Cleanup temp files and dir
        print("Cleaning up temporary models...")
        for fname in model_files:
            fpath = os.path.join(tmp_path, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        os.rmdir(tmp_path)

        #
        # Compose the results
        out = TrainResults(last_model=last_model,
                           best_model=best_model, best_epoch=best_epoch,
                           train_history=train_itr_history)

        return out

    def get_input_size(self) -> Tuple[int, int, int]:
        """Return the size of the input layer as tuple for (width, height, depth)."""

        l0 = self._model.get_layer(index=0)  # type: keras.engine.input_layer.InputLayer
        # Get the dimension (the first number is the sample number)
        _, w, h, depth = l0.input_shape
        return w, h, depth

    def get_output_size(self) -> int:
        out_layer = self._model.layers[-1]
        return out_layer.units

    def generate_heatmap(self, image: Image, method: str='gradcam', **kwargs) -> Tuple[Image, Image, Image]:
        """

        :param image:
        :param method: One among {'gradcam', 'rise'}
        :param kwargs: The specific arguments for each method
        For 'gradcam': 'layer_name:str' is the name of the convolutional layer to analyze. Normally the last alyer gives best results. E.g., for VGG16 try 'layer_name="block5_conv3"'.
        For 'rise': 'N' is the number of iterations (and needed predictions), 's' is ???, 'p1' is ???
        :return: a 3-tuple with i) the saliency map (as greyscale image) ii) the heatmap, iii) the original image, scaled and overlapped with the heatmap.
        """

        import timl.xai.rise

        #
        # Converts the input image to the numpy format needed by the model
        # Converts the image into a numpy array, already scaled for the model.
        scaled_img_sample_np = Classifier.img_to_numpy_sample(image=image, model=self._model)

        if method == 'gradcam':
            import timl.xai.gradcam as gradcam
            layer_name = kwargs.get('layer_name', 'last_convolution')

            if self._gracam_modified_model is None:
                self._init_gradcam_model()

            assert self._gracam_modified_model is not None

            #
            # Get the saliency (greyscale) maps from the XAI method
            # greymap, heatmap, composite = generate_xai_maps(model=self._model, image=img_np, layer_name=layer_name)
            # timl.xai.gradcam.generate_saliency_map () method is modified and a new parameter is added to
            # multple mdel load during loop call of heat-map generation
            greymap = gradcam.generate_saliency_map(model=self._model,
                                                    linear_model=self._gracam_modified_model,
                                                    image=scaled_img_sample_np,
                                                    layer_name=layer_name)

        elif method == 'rise':
            n_iterations = kwargs.get('N', 2000)
            mask_size = kwargs.get('s', 6)

            greymap = timl.xai.rise.generate_saliency_map(model=self._model, image=scaled_img_sample_np,
                                                          N=n_iterations, s=mask_size)
        else:
            raise Exception("Unknown method '{}' for generating heatmaps".format(method))

        #
        # Generate the heatmap from the greymap
        # TODO -- get rid of cv2 dependency!
        # See: https://stackoverflow.com/questions/43457308/is-there-any-good-color-map-to-convert-gray-scale-image-to-colorful-ones-using-p
        # Grab the colormap (a lookup table) and hard-code it into our code.
        import cv2
        heatmap = cv2.applyColorMap(np.uint8(255 * greymap), cv2.COLORMAP_JET)
        # Image is generated as BGR
        # Go back to RGB mode
        heatmap = np.flip(heatmap, axis=2)
        # Re-normalize in [0-1]
        heatmap = np.float32(heatmap) / 255.0
        assert heatmap.dtype == np.float32

        #
        # Compose colormap and original image
        scaled_img_np = scaled_img_sample_np[0]
        composed_img = heatmap + np.float32(scaled_img_np)
        # Re-normalize in [0-1]
        composed_img = composed_img / np.max(composed_img)
        assert composed_img.dtype == np.float32

        #
        # Convert all numpy images in PIL.Image.Image instances
        greymap_pil = PIL.Image.fromarray(np.uint8(255.0 * greymap), 'L')
        heatmap_pil = PIL.Image.fromarray(np.uint8(255.0 * heatmap), 'RGB')
        composite_pil = PIL.Image.fromarray(np.uint8(255.0 * composed_img), 'RGB')

        return greymap_pil, heatmap_pil, composite_pil

    def predict(self, generator: SkincareDataGenerator) -> np.ndarray:
        """Performs an image-by-image prediction.
         For each image will return inferred scores (probability distribution)"""

        # Get the p distributions
        y = self._model.predict_generator(generator, verbose=True)

        assert y.shape[0] == generator.get_n_images()
        assert y.shape[1] > 1

        return y

    def predict_image(self, image: Image) -> Tuple[List[float], List[float]]:
        """
        Computes the predictions of an image over all the target classes
        together with a confidence of each prediction.

        :param image:
        :return: Two lists of floats: predictions and confidences.
        Both lists' length equals to the number of classes predicted by the model.
        predictions is the probability distribution, the output of the softmax, summing to 1.0.
        confidences gives a confidence level, in range [-1,1] of the corresponding prediction.
         Consider as baseline b the uniform distribution threshold b=1/n_classes.
         Given a prediction p, the confidence is the fraction of p over the interval [b,1.0].
         It is computed as negative if the p is below b, as the fraction in [b,0.0]
        """

        x_data = Classifier.img_to_numpy_sample(image=image, model=self._model)

        prediction = self._model.predict(x=x_data)
        first_prediction = prediction[0]
        # Convert to a standard Python array
        out_array = [float(p) for p in first_prediction]

        #
        # Compute confidence
        n_predictions = len(out_array)
        b = 1.0 / n_predictions
        upper_interval = 1.0 - b

        confidence = [(p - b) / upper_interval if p > b else (p - b) / b for p in out_array]

        return out_array, confidence

    @staticmethod
    def img_to_numpy_sample(image: Image, model: Model) -> np.ndarray:
        """
        Converts a PIL image into a numpy array, normalizing the pixels into the range 0-1.
        The image is scaled to the input of the given model.

        :param model: The model, used to infer the images size
        :param image: The input Image
        :return: the 4-dimensional np.ndarray [samples(1), width, height, depth] containing the image data
        """

        #
        # Analyze the model to get what resolution it wants
        # Get the first (input) layer
        l0 = model.get_layer(index=0)  # type: keras.engine.input_layer.InputLayer
        # Get the dimension (the first number is the sample number)
        _, w, h, depth = l0.input_shape
        img_size = (w, h)

        # Resize (invert the size indices)
        # print("Resizing the input image to {}x{}".format(w, h))
        image = image.resize(img_size[::-1], PIL.Image.NEAREST)

        # Prepare space for 1 image, given size, given depth (most likely 3: RGB)
        # (It is a 4-dimension vector, but first dimension size is 1, because it is 1 sample)
        x_data = np.zeros((1, *img_size, depth))
        # convert the image into an numpy array and store it at the first (and only) position
        x_data[0, ] = np.array(image, dtype=np.float32)
        x_data /= 255.0  # Normalize

        assert len(x_data.shape) == 4
        assert len(x_data) == 1

        return x_data

    @staticmethod
    def classify_image(model: Model, image: Image) -> List[float]:

        x_data = Classifier.img_to_numpy_sample(image=image, model=model)

        prediction = model.predict(x=x_data)
        first_prediction = prediction[0]
        # Convert to a standard Python array
        out_array = [float(p) for p in first_prediction]

        return out_array

    @staticmethod
    def classify(model: Model, image_path: str) -> List[float]:
        # Load the image from disk
        img = PIL.Image.open(image_path)
        return Classifier.classify_image(model=model, image=img)

