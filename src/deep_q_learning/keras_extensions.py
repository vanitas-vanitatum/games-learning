import keras.backend as K
from tensorflow.python.keras.utils import to_categorical
import tensorflow as tf

BOARD_SIZE = 9

# def smooth_loss(args):
#     y_true, y_pred, action_indices = args
#     action_indices = K.one_hot(action_indices, BOARD_SIZE)
#     y_pred = K.sum(y_pred * action_indices, axis=1)
#     diff = y_true - y_pred
#     abs_diff = K.abs(diff)
#     positive_mask = K.cast(abs_diff <= 1, K.floatx())
#
#     loss = 0.5 * K.pow(diff, 2) * positive_mask + (abs_diff - 0.5) * (1 - positive_mask)
#     return K.mean(K.sum(loss, axis=1))

def mse_loss(args):
    y_true, y_pred, action_indices = args
    action_indices = K.reshape(tf.one_hot(action_indices, BOARD_SIZE), (-1, BOARD_SIZE))
    y_pred = K.reshape(tf.reduce_sum(tf.multiply(y_pred, action_indices), axis=1), (-1, 1))
    diff = y_true - y_pred
    sqr_diff = tf.square(diff)
    return tf.reduce_mean(sqr_diff)


def mean_squared_loss(args):
    y_true, y_pred, action_indices = args
    y_pred = K.gather(y_pred, action_indices)
    diff = K.pow(y_true - y_pred, 2)
    return K.mean(K.sum(diff, axis=1))
