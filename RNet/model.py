import tensorflow as tf
from tensorflow.keras import Input, Model, utils
from tensorflow.keras.layers import Layer, Conv2D, PReLU, MaxPool2D, Lambda, Flatten, Dense

def square(bboxes):
    """
    Turn the bounding boxes into squares.
    Args:
    - bboxes: (n, 4) tensor with each row encoded as [y1, x1, y2, x2]
    Return:
    - square_bboxes: (n, 4) tensor of squared bounding boxes
    """
    h = bboxes[:, 2] - bboxes[:, 0]
    w = bboxes[:, 3] - bboxes[:, 1]
    size = tf.math.maximum(h, w)
    x_margin = (size - w) / 2
    y_margin = (size - h) / 2
    y1 = bboxes[:, 0] - y_margin
    x1 = bboxes[:, 1] - x_margin
    y2 = bboxes[:, 2] + y_margin
    x2 = bboxes[:, 3] + x_margin
    square_bboxes = tf.stack([y1, x1, y2, x2], axis=1)
    return square_bboxes

def calibrate_and_nms(input_tensors):
    """
    This function will operate at the end of the model when predicting.
    The codes are entirely written using tensorflow.
    """

    # Parse the input
    class_scores = input_tensors[0][:, 0] # shape (n,)
    bbox_scores = input_tensors[1] # shape (n, 4)
    origins = input_tensors[2] # shape (n, 3)

    # Positive windows' indices (predicted of course)
    pos_indices = tf.where(class_scores >= 0.5)

    # Get the predicted bounding boxes of the positive windows
    pred_bboxes = tf.gather_nd(bbox_scores, pos_indices)
    origins = tf.gather_nd(origins, pos_indices)

    # pred_bboxes = square(pred_bboxes)

    # Get the calibrated windows' confidence
    confidence = tf.gather_nd(class_scores, pos_indices)

    # Convert to original coordinates
    y1_x1 = origins[:, :2]
    original_size = tf.reshape(origins[:, 2], [-1, 1])
    original_bboxes_y1_x1 = y1_x1 + pred_bboxes[:, :2] * original_size 
    original_bboxes_y2_x2 = y1_x1 + pred_bboxes[:, 2:] * original_size
    original_bboxes = tf.concat(values=[original_bboxes_y1_x1, original_bboxes_y2_x2], axis=1)

    # Non max suppression
    selected_indices = tf.image.non_max_suppression(original_bboxes, confidence, tf.shape(confidence)[0], 0.7)
    selected_bboxes = tf.gather(original_bboxes, selected_indices)

    # Calibrate
    calibrated_bboxes = square(selected_bboxes)
    # calibrated_bboxes = selected_bboxes

    # Round out everything
    calibrated_bboxes = tf.cast(calibrated_bboxes, tf.int32)

    return calibrated_bboxes

def pad_0(num_neurons):
    def pad(input_tensor):
        batch_size = tf.shape(input_tensor)[0]
        return tf.concat(values=[tf.zeros(shape=(batch_size, 1)), input_tensor], axis=-1)
    return pad

def rnet():
    inputs = Input(shape=(24, 24, 3), name='inputs')
    conv_1 = Conv2D(filters=28, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_1')(inputs)
    prelu_1 = PReLU(name='prelu_1')(conv_1)
    pool_1 = MaxPool2D(name='pool_1')(prelu_1)
    conv_2 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer='he_normal', name='conv_2')(pool_1)
    prelu_2 = PReLU(name='prelu_2')(conv_2)
    pool_2 = MaxPool2D(pool_size=(3, 3), strides=2, name='pool_2')(prelu_2)
    conv_3 = Conv2D(filters=64, kernel_size=(2, 2), kernel_initializer='he_normal', name='conv_3')(pool_2)
    prelu_3 = PReLU(name='prelu_3')(conv_3)
    flatten_1 = Flatten(name='flatten_1')(prelu_3)
    dense_1 = Dense(units=128, kernel_initializer='he_normal', name='dense_1')(flatten_1)
    prelu_4 = PReLU(name='prelu_4')(dense_1)
    output_1 = Dense(units=2, activation='softmax', kernel_initializer='lecun_normal', name='output_1')(prelu_4)
    padded_output_1 = Lambda(function=pad_0(num_neurons=2), name='class')(output_1)
    output_2 = Dense(units=4, kernel_initializer='lecun_normal', name='output_2')(prelu_4)
    padded_output_2 = Lambda(function=pad_0(num_neurons=4), name='bbox')(output_2)
    output_3 = Dense(units=10, kernel_initializer='lecun_normal', name='output_3')(prelu_4)
    padded_output_3 = Lambda(function=pad_0(num_neurons=10), name='landmark')(output_3)
    model = Model(inputs=inputs, outputs=[padded_output_1, padded_output_2, padded_output_3])
    return model

def rnet_predict():
    origins = Input(shape=(3), name='origins')
    inputs = Input(shape=(24, 24, 3), name='inputs')
    conv_1 = Conv2D(filters=28, kernel_size=(3, 3), kernel_initializer='zeros', name='conv_1')(inputs)
    prelu_1 = PReLU(name='prelu_1')(conv_1)
    pool_1 = MaxPool2D(name='pool_1')(prelu_1)
    conv_2 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer='zeros', name='conv_2')(pool_1)
    prelu_2 = PReLU(name='prelu_2')(conv_2)
    pool_2 = MaxPool2D(pool_size=(3, 3), strides=2, name='pool_2')(prelu_2)
    conv_3 = Conv2D(filters=64, kernel_size=(2, 2), kernel_initializer='zeros', name='conv_3')(pool_2)
    prelu_3 = PReLU(name='prelu_3')(conv_3)
    flatten_1 = Flatten(name='flatten_1')(prelu_3)
    dense_1 = Dense(units=128, kernel_initializer='zeros', name='dense_1')(flatten_1)
    prelu_4 = PReLU(name='prelu_4')(dense_1)
    output_1 = Dense(units=2, activation='softmax', kernel_initializer='zeros', name='output_1')(prelu_4)
    output_2 = Dense(units=4, kernel_initializer='zeros', name='output_2')(prelu_4)
    output = Lambda(function=calibrate_and_nms, name='calibrate_and_nms')([output_1, output_2, origins])
    model = Model(inputs=[inputs, origins], outputs=output)
    return model