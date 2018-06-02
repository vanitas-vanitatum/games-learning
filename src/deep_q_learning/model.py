import keras.backend as K
from keras.layers import Conv2D, Flatten, Input, Dense, Lambda, Add, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2

from src.deep_q_learning.keras_extensions import mean_squared_loss


def get_atari_model():
    inputs = Input(shape=(None, None, 4))
    conv = Conv2D(32, 8, strides=(4, 4), activation='relu')(inputs)
    conv = Conv2D(64, 4, strides=(2, 2), activation='relu')(conv)
    conv = Conv2D(64, 3, activation='relu')(conv)
    flat = Flatten()(conv)


def get_model_3x3():
    inputs = Input(shape=(9,), name='input')
    action_input = Input(shape=(1,), name='actions', dtype='int32')
    expected_state_values = Input(shape=(1,), name='expected')
    layer = Dense(128, kernel_regularizer=l2(0.0001))(inputs)
    layer = LeakyReLU()(layer)
    layer = Dense(96, kernel_regularizer=l2(0.0001))(layer)
    layer = LeakyReLU()(layer)
    output = Dense(9, name='output', activation='tanh', kernel_regularizer=l2(0.0001))(layer)
    last = Lambda(mean_squared_loss, output_shape=(1,), name='loss')([expected_state_values, output, action_input])

    model = Model([inputs, expected_state_values, action_input], [last])
    model.compile(RMSprop(lr=0.001, clipvalue=5), loss=lambda y_true, y_pred: y_pred)
    model.summary()
    predict_function = K.function([inputs], [output])
    return model, predict_function


def create_actor_model():
    state_input = Input(shape=(9,), name='input')
    layer = Dense(128, kernel_regularizer=l2(0.0001))(state_input)
    layer = LeakyReLU()(layer)
    layer = Dense(96, kernel_regularizer=l2(0.0001))(layer)
    layer = LeakyReLU()(layer)
    output = Dense(9, name='output', activation='relu', kernel_regularizer=l2(0.0001))(layer)

    model = Model(inputs=[state_input], outputs=[output])
    return state_input, model


def create_critic_model():
    state_input = Input(shape=(9,), name='input')
    state_h1 = Dense(128, activation='relu')(state_input)
    state_h2 = Dense(48)(state_h1)

    action_input = Input(shape=(9,), name='action_input')
    action_h1 = Dense(48)(action_input)

    merged = Add()([state_h2, action_h1])
    merged_h1 = Dense(24, activation='relu')(merged)
    output = Dense(1, activation='relu')(merged_h1)
    model = Model(inputs=[state_input, action_input], outputs=[output])
    return state_input, action_input, model


if __name__ == '__main__':
    get_model_3x3()
