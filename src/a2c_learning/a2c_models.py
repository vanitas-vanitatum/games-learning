from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l2

INPUT_SIZE = 9
NB_ACTIONS = 9


def create_policy_network():
    input_tensor = Input(shape=(INPUT_SIZE,), name='input')
    layer = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(input_tensor)
    # layer = Dense(32, activation='relu')(layer)
    output = Dense(9, activation='softmax', name='output')(layer)
    model = Model(inputs=[input_tensor], outputs=[output])
    return model, input_tensor


def create_value_network():
    input_tensor = Input(shape=(INPUT_SIZE,), name='input')
    layer = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(input_tensor)
    # layer = Dense(32, activation='relu')(layer)
    output = Dense(1)(layer)
    model = Model(inputs=[input_tensor], outputs=[output])
    return model, input_tensor
