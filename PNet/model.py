import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Conv2D, PReLU, MaxPool2D, Lambda

def pad_0(num_neurons):
    def pad(input_tensor):
        batch_size = tf.shape(input_tensor)[0]
        return tf.concat(values=[tf.zeros(shape=(batch_size, 1)), tf.reshape(tensor=input_tensor, shape=(batch_size, num_neurons))], axis=-1)
    return pad

def calibrate_and_nms(input_tensors):
    """
    This function will operate at the end of the model when predicting.
    The codes are entirely written using tensorflow.
    """

    # Parse the input
    class_scores = input_tensors[0][0, :, :, 0] # shape (n,)
    bbox_scores = input_tensors[1][0] # shape (n, 4)

    # Positive windows' indices (predicted of course)
    pos_indices = tf.where(class_scores >= 0.5)

    # Get the predicted bounding boxes of the positive windows
    pred_bboxes = tf.gather_nd(bbox_scores, pos_indices)

    # Calibrate
    size = tf.math.reduce_max(pred_bboxes[:, 2:], axis=1, keepdims=True)
    delta = size - pred_bboxes[:, 2:]
    calibrated_bboxes_x1_y1 = pred_bboxes[:, :2] - delta / 2
    calibrated_bboxes_y1_x1 = calibrated_bboxes_x1_y1[:, ::-1] * 12 + tf.cast(pos_indices, tf.float32) * 2
    calibrated_bboxes_y2_x2 = calibrated_bboxes_y1_x1 + size * 12
    calibrated_bboxes = tf.concat([calibrated_bboxes_y1_x1, calibrated_bboxes_y2_x2], axis=1)

    # Get the calibrated windows' confidence
    confidence = tf.gather_nd(class_scores, pos_indices)

    # Non max suppression
    selected_indices = tf.image.non_max_suppression(calibrated_bboxes, confidence, tf.shape(confidence)[0], 0.5)
    selected_bboxes = tf.gather(calibrated_bboxes, selected_indices)
    selected_confidence = tf.gather(confidence, selected_indices)

    return selected_bboxes, selected_confidence

def pnet():
    inputs = Input(shape=(12, 12, 3), name='inputs')
    conv_1 = Conv2D(filters=10, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_1')(inputs)
    prelu_1 = PReLU(shared_axes=[1, 2], name='prelu_1')(conv_1)
    pool_1 = MaxPool2D(name='pool_1')(prelu_1)
    conv_2 = Conv2D(filters=16, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_2')(pool_1)
    prelu_2 = PReLU(shared_axes=[1, 2], name='prelu_2')(conv_2)
    conv_3 = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_3')(prelu_2)
    prelu_3 = PReLU(shared_axes=[1, 2], name='prelu_3')(conv_3)
    output_1 = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax', kernel_initializer='lecun_normal', name='output_1')(prelu_3)
    padded_output_1 = Lambda(function=pad_0(num_neurons=2), name='class')(output_1)
    output_2 = Conv2D(filters=4, kernel_size=(1, 1), kernel_initializer='lecun_normal', name='output_2')(prelu_3)
    padded_output_2 = Lambda(function=pad_0(num_neurons=4), name='bbox')(output_2)
    output_3 = Conv2D(filters=10, kernel_size=(1, 1), kernel_initializer='lecun_normal', name='output_3')(prelu_3)
    padded_output_3 = Lambda(function=pad_0(num_neurons=10), name='landmark')(output_3)
    model = Model(inputs=inputs, outputs=[padded_output_1, padded_output_2, padded_output_3])
    return model

def pnet_predict():
    inputs = Input(shape=(None, None, 3), name='inputs')
    conv_1 = Conv2D(filters=10, kernel_size=(3, 3), kernel_initializer='zeros', name='conv_1')(inputs)
    prelu_1 = PReLU(shared_axes=[1, 2], name='prelu_1')(conv_1)
    pool_1 = MaxPool2D(name='pool_1')(prelu_1)
    conv_2 = Conv2D(filters=16, kernel_size=(3, 3), kernel_initializer='zeros', name='conv_2')(pool_1)
    prelu_2 = PReLU(shared_axes=[1, 2], name='prelu_2')(conv_2)
    conv_3 = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='zeros', name='conv_3')(prelu_2)
    prelu_3 = PReLU(shared_axes=[1, 2], name='prelu_3')(conv_3)
    output_1 = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax', kernel_initializer='zeros', name='output_1')(prelu_3)
    output_2 = Conv2D(filters=4, kernel_size=(1, 1), kernel_initializer='zeros', name='output_2')(prelu_3)
    output = Lambda(function=calibrate_and_nms, name='calibrate_and_nms')([output_1, output_2])
    model = Model(inputs=inputs, outputs=output)
    return model