"""
Visualization of heat map using rise algorithm from
url: http://cs-people.bu.edu/vpetsiuk/rise/index.html
@author: duynhm
"""

import numpy as np
from skimage.transform import resize
from keras import Model
import keras.backend as K

from typing import Tuple

# This is the batch size for the prediction using different masks
# A high value might lead to out-of-memory errors.
BATCH_SIZE = 100


def _generate_masks(N, s, p1, img_size: Tuple[int, int]):
    w, h = img_size

    cell_size = np.ceil(np.array(img_size) / s)
    up_size = (s + 1) * cell_size

    # cell_size = np.ceil(np.array(w, h) / s)

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, w, h))

    # Generating counting
    for i in range(N):
        # print("RISE generating masks step " + str(i))

        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + w, y:y + h]
    masks = masks.reshape(-1, w, h, 1)
    return masks


def _explain(model: Model, img_size: Tuple[int, int], inp, masks, N, p1):
    preds = []
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    for i in range(0, N, BATCH_SIZE):
        # print("RISE explain step " + str(i))
        preds.append(model.predict(masked[i:min(i + BATCH_SIZE, N)]))
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, img_size[0], img_size[1])
    sal = sal / N / p1
    return sal


def _predictions_to_class(image: np.ndarray, model: Model):
    y_pred = model.predict(image)
    # print("Rise prediction: ", y_pred)
    class_idx = np.argmax(y_pred)
    return class_idx


def generate_saliency_map(model: Model, image: np.ndarray, N, s, p1=0.5) -> np.ndarray:
    """
    Generates a grescale saliency map using the RISE method:
    <http://cs-people.bu.edu/vpetsiuk/rise/index.html>

    :param model: The pre-trained convolutional model
    :param image: The input image, as 1-sample of shape (1, w, h, 3)
    :param N: Number of iterations (i.e., masks applications.
    Beware, this is also the number of times the model will be invoked for predictions!)
    :param s:
    :param p1:
    :return:
    """

    if not (image.dtype == np.float32 or image.dtype == np.float64):
        raise Exception("Image type must be float32 or float64")

    # Expecting image.shape to be (1, ?, ?, 3)
    if len(image.shape) != 4:
        raise Exception("Image array needs 4 dimensions (1-sample, w, h, d)")

    img_min = np.min(image)
    img_max = np.max(image)
    if img_min < 0.0 or img_max > 1.0:
        raise Exception("Image must be provided in normalized range [0,1]. Found min/max {}/{}".format(img_min, img_max))

    img_w, img_h = image.shape[1:3]

    # Run Explanations
    masks = _generate_masks(N=N, s=s, p1=p1, img_size=(img_w, img_h))
    sal = _explain(model=model, img_size=(img_w, img_h), inp=image, masks=masks, N=N, p1=p1)
    idx = _predictions_to_class(image=image, model=model)
    class_idx = idx

    # Grey map
    grey_map = sal[class_idx]
    arr_min, arr_max = np.min(grey_map), np.max(grey_map)
    grey_map = (grey_map - arr_min) / (arr_max - arr_min + K.epsilon())

    # The width and height of the grey map are the same as the ones of the original image.
    # print(image.shape, grey_map.shape)
    assert image.shape[1:3] == grey_map.shape

    return grey_map
