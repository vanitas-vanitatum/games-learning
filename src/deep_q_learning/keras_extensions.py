import keras.backend as K
import tensorflow as tf


def smooth_loss(args):
    y_true, y_pred, action_indices = args
    y_pred = K.gather(y_pred, action_indices)
    diff = y_pred - y_true
    abs_diff = K.abs(diff)
    positive_mask = K.cast(abs_diff <= 1, K.floatx())

    loss = 0.5 * K.square(diff) * positive_mask + (abs_diff - 0.5) * (1 - positive_mask)
    return K.mean(loss)


def mean_squared_loss(args):
    y_true, y_pred, action_indices = args
    one_actions = tf.one_hot(action_indices, depth=9)
    summed_y_pred = K.sum(y_pred * one_actions, axis=1)

    abs_pred = K.softmax(y_pred)

    diff = K.square(summed_y_pred - y_true)
    # return K.mean(diff) + 1 / 10 * K.categorical_crossentropy(abs_pred, abs_pred)
    return K.mean(diff)
