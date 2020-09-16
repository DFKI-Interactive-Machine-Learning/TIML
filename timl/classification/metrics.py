# Extra metric functions for the keras framework

# When adding a metric here, please also add it to the Classifier._load_model()
# to allow for proper saving/loading of models


def old_mean_f1_score(y_true, y_pred):
    """Computation of the f1-score for problems of multi-label classification.
    Input tensors are meant to be of shape (m, n_labels).
    We compute the f1 score for each sample and then average them."""

    import keras.backend as K

    # Adapted from
    # https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-1)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    mean_f1_val = K.mean(f1_val)  # Compute the mean. Output is a scalar
    return mean_f1_val


# https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    import tensorflow as tf

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=-1)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=-1)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=-1)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels

    return macro_cost


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    import tensorflow as tf

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


#
#
# Fixed from:
# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

def mean_f1_score(y_true, y_pred):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """

    import keras.backend as K
    import tensorflow as tf

    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=-1)
    # tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=-1)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=-1)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=-1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def loss_1_minus_f1(y_true, y_pred):
    import keras.backend as K
    import tensorflow as tf

    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=-1)
    # tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=-1)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=-1)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=-1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

#
#
#


def loss_1mf1_by_bce(y_true, y_pred):
    import keras.backend as K

    loss_f1 = loss_1_minus_f1(y_true, y_pred)
    bce = K.binary_crossentropy(target=y_true, output=y_pred, from_logits=False)

    return loss_f1 * bce



def loss_binary_cross_entropy(y_true, y_pred):
    """This is the function to compute the binary cross entropy for multi-label problems.
    both y_true and y_pred have shape [m, n_classes]"""
    # yt = y true, yp = y predicted
    # L(yt, yp) =  - 1/n * sum(yt * log(yp) + (1 - yt) * log(1 - yp))

    import keras.backend as K

    t = y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred)
    return - K.mean(t)


#
# From Keras docs in the implementation of sigmoid_cross_entropy_with_logits:
#
#  For brevity, let `x = logits`, `z = labels`.
#
# Hence, to ensure stability and avoid overflow, the implementation uses this equivalent formulation
#       max(x, 0) - x * z + log(1 + exp(-abs(x)))
def loss_alt_binary_cross_entropy(y_true, y_pred_logits):
    import keras.backend as K
    return K.max(y_pred_logits, 0) - y_pred_logits * y_true + K.log(1 + K.exp(- K.abs(y_pred_logits)))
