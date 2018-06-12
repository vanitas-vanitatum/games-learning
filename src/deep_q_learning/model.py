from keras.layers import Conv2D, Flatten, Input, Dense, Lambda, Dropout, BatchNormalization, Activation
from keras.models import Model
from src.deep_q_learning.keras_extensions import mse_loss, mean_squared_loss

import keras.backend as K


def get_atari_model():
    inputs = Input(shape=(None, None, 4))
    conv = Conv2D(32, 8, strides=(4, 4), activation='relu')(inputs)
    conv = Conv2D(64, 4, strides=(2, 2), activation='relu')(conv)
    conv = Conv2D(64, 3, activation='relu')(conv)
    flat = Flatten()(conv)


def get_model_3x3():
    inputs = Input(shape=(10,), name='input')
    action_input = Input(shape=(1,), name='actions', dtype='int32')
    expected_state_values = Input(shape=(1,), name='expected')
    layer = Dense(128)(inputs)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(96, activation='relu')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    output = Dense(9, name='output')(layer)
    last = Lambda(mse_loss, output_shape=(1,), name='loss')([expected_state_values, output, action_input])

    model = Model([inputs, expected_state_values, action_input], [last])
    model.compile('adam', loss=lambda y_true, y_pred: y_pred)
    model.summary()
    predict_function = K.function([inputs], [output])
    return model, predict_function


def get_model_simple():
    inputs = Input(shape=(9,), name='input')
    action_input = Input(shape=(1,), name='actions', dtype='int32')
    expected_state_values = Input(shape=(1,), name='expected')
    layer = Dense(100)(inputs)
    layer = Activation('relu')(layer)
    #layer = Dropout(0.5)(layer)
    #layer = Dense(96, activation='relu')(layer)
    #layer = Activation('relu')(layer)
    #layer = Dropout(0.5)(layer)
    output = Dense(9, name='output', activation='tanh')(layer)
    last = Lambda(mse_loss, output_shape=(1,), name='loss')([expected_state_values, output, action_input])

    model = Model([inputs, expected_state_values, action_input], [last])
    model.compile('adam', loss=lambda y_true, y_pred: y_pred)
    model.summary()
    predict_function = K.function([inputs], [output])
    return model, predict_function

if __name__ == '__main__':
    get_model_3x3()
