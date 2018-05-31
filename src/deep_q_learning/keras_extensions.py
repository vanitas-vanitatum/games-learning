import keras.backend as K


def smooth_loss(args):
    y_true, y_pred, action_indices = args
    y_pred = K.gather(y_pred, action_indices)
    diff = y_true - y_pred
    abs_diff = K.abs(diff)
    positive_mask = K.cast(abs_diff <= 1, K.floatx())

    loss = 0.5 * K.pow(diff, 2) * positive_mask + (abs_diff - 0.5) * (1 - positive_mask)
    return K.mean(K.sum(loss, axis=1))


def mean_squared_loss(args):
    y_true, y_pred, action_indices = args
    y_pred = K.gather(y_pred, action_indices)
    diff = 0.5 * K.pow(y_true - y_pred, 2)
    return K.mean(K.sum(diff, axis=1))
