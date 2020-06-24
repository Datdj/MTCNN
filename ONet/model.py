import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Conv2D, PReLU, MaxPool2D, Lambda, Flatten, Dense

def pad_0(num_neurons):
    def pad(input_tensor):
        batch_size = tf.shape(input_tensor)[0]
        return tf.concat(values=[tf.zeros(shape=(batch_size, 1)), input_tensor], axis=-1)
    return pad

def onet():
    inputs = Input(shape=(48, 48, 3), name='inputs')
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_1')(inputs)
    prelu_1 = PReLU(name='prelu_1')(conv_1)
    pool_1 = MaxPool2D(name='pool_1')(prelu_1)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_2')(pool_1)
    prelu_2 = PReLU(name='prelu_2')(conv_2)
    pool_2 = MaxPool2D(pool_size=(3, 3), strides=2, name='pool_2')(prelu_2)
    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_3')(pool_2)
    prelu_3 = PReLU(name='prelu_3')(conv_3)
    pool_3 = MaxPool2D(name='pool_3')(prelu_3)
    conv_4 = Conv2D(filters=128, kernel_size=(2, 2), kernel_initializer='he_normal', name='conv_4')(pool_3)
    prelu_4 = PReLU(name='prelu_4')(conv_4)
    flatten_1 = Flatten(name='flatten_1')(prelu_4)
    dense_1 = Dense(units=256, kernel_initializer='he_normal', name='dense_1')(flatten_1)
    prelu_5 = PReLU(name='prelu_5')(dense_1)
    output_1 = Dense(units=2, activation='softmax', kernel_initializer='lecun_normal', name='output_1')(prelu_5)
    padded_output_1 = Lambda(function=pad_0(num_neurons=2), name='class')(output_1)
    output_2 = Dense(units=4, kernel_initializer='lecun_normal', name='output_2')(prelu_5)
    padded_output_2 = Lambda(function=pad_0(num_neurons=4), name='bbox')(output_2)
    output_3 = Dense(units=10, kernel_initializer='lecun_normal', name='output_3')(prelu_5)
    padded_output_3 = Lambda(function=pad_0(num_neurons=10), name='landmark')(output_3)
    model = Model(inputs=inputs, outputs=[padded_output_1, padded_output_2, padded_output_3])
    return model