"""
Visualization of heat map using grad cam algorithm
url: https://fairyonice.github.io/Grad-CAM-with-keras-vis.html

"""

import keras
import numpy as np
from vis.utils import utils
import keras.backend as K
from scipy.ndimage.interpolation import zoom

from typing import Optional


def generate_saliency_map(model: keras.Model, linear_model: keras.Model, image: np.ndarray, layer_name: str,
                          class_idx: Optional[int] = None) -> np.ndarray:
    """

    :param model: The keras model which will process the image.
    :param image: The ndarray of the image. Size (1, w, h, depth), where depth in normally 3 (RGB)
    :param layer_name: The name of the layer from which to extract the gradients and the heatmap
    :param class_idx: The claas number for which the gradcam should run. If none, the argmax of the prediction is used.
    :return: The saliency map of the input image. Resolution depends on the layer shape.
             Number of channels is 1, with values in range [0,1].
             The output type is np.float32.
    """
    if not (image.dtype == np.float32 or image.dtype == np.float64):
        raise Exception("Image type must be float32 or float64")

    img_min = np.min(image)
    img_max = np.max(image)
    if img_min < 0.0 or img_max > 1.0:
        raise Exception("Image must be provided in normalized range [0,1]. Found min/max {}/{}".format(img_min, img_max))

    if class_idx is None:
        y_pred = model.predict(image)
        class_idx = np.argmax(y_pred)

    #layer_idx = utils.find_layer_idx(model, 'predictions')
    # Swap softmax with linear
    #model.layers[layer_idx].activation = keras.activations.linear
    #model = utils.apply_modifications(model)
    model = linear_model
    final_fmap_index = utils.find_layer_idx(model, layer_name)
    penultimate_output = model.layers[final_fmap_index].output
    layer_input = model.input
    # This model must already use linear activation for the final layer
    loss = model.layers[-1].output[..., class_idx]
    grad_wrt_fmap = K.gradients(loss, penultimate_output)[0]

    # Create function that evaluate the gradient for a given input
    # This function accept numpy array
    grad_wrt_fmap_fn = K.function([layer_input, K.learning_phase()],
                                  [penultimate_output, grad_wrt_fmap])

    ## evaluate the derivative_fn
    fmap_eval, grad_wrt_fmap_eval = grad_wrt_fmap_fn([image, 0])

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grad_wrt_fmap_eval /= (np.max(grad_wrt_fmap_eval) + K.epsilon())

    # print(grad_wrt_fmap_eval.shape)
    alpha_k_c = grad_wrt_fmap_eval.mean(axis=(0, 1, 2)).reshape((1, 1, 1, -1))
    Lc_Grad_CAM = np.maximum(np.sum(fmap_eval * alpha_k_c, axis=-1), 0).squeeze()

    ## upsampling the class activation map to the size of the input image
    scale_factor = np.array(image.shape[1:3]) / np.array(Lc_Grad_CAM.shape)
    _grad_CAM = zoom(Lc_Grad_CAM, scale_factor)
    ## normalize to range between 0 and 1
    arr_min, arr_max = np.min(_grad_CAM), np.max(_grad_CAM)
    grad_CAM = (_grad_CAM - arr_min) / (arr_max - arr_min + K.epsilon())

    assert grad_CAM.dtype == np.float32

    return grad_CAM
