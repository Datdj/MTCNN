import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Conv2D, PReLU, MaxPool2D, Lambda

def pad_0(num_neurons, batch_size):
    def pad(input_tensor):
        return tf.concat(values=[tf.zeros(shape=(batch_size, 1)), tf.reshape(tensor=input_tensor, shape=(batch_size, num_neurons))], axis=-1)
    return pad

def pnet(batch_size):
    inputs = Input(shape=(12, 12, 3), name='inputs')
    conv_1 = Conv2D(filters=10, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_1')(inputs)
    prelu_1 = PReLU(name='prelu_1')(conv_1)
    pool_1 = MaxPool2D(name='pool_1')(prelu_1)
    conv_2 = Conv2D(filters=16, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_2')(pool_1)
    prelu_2 = PReLU(name='prelu_2')(conv_2)
    conv_3 = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_3')(prelu_2)
    prelu_3 = PReLU(name='prelu_3')(conv_3)
    output_1 = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax', kernel_initializer='lecun_normal', name='output_1')(prelu_3)
    padded_output_1 = Lambda(function=pad_0(num_neurons=2, batch_size=batch_size), name='class')(output_1)
    output_2 = Conv2D(filters=4, kernel_size=(1, 1), kernel_initializer='lecun_normal', name='output_2')(prelu_3)
    padded_output_2 = Lambda(function=pad_0(num_neurons=4, batch_size=batch_size), name='bbox')(output_2)
    output_3 = Conv2D(filters=10, kernel_size=(1, 1), kernel_initializer='lecun_normal', name='output_3')(prelu_3)
    padded_output_3 = Lambda(function=pad_0(num_neurons=10, batch_size=batch_size), name='landmark')(output_3)
    model = Model(inputs=inputs, outputs=[padded_output_1, padded_output_2, padded_output_3])
    return model