from tensorflow.math import reduce_sum
from tensorflow.keras.losses import binary_crossentropy, MSE

def BCE_with_sample_type_indicator(y_true, y_pred): 
    return reduce_sum(y_true[:, 0] * binary_crossentropy(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]))

def MSE_with_sample_type_indicator(y_true, y_pred):
    return reduce_sum(y_true[:, 0] * MSE(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]))