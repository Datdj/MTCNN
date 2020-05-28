import tensorflow as tf
from tensorflow.math import reduce_sum
from tensorflow.keras.losses import binary_crossentropy, MSE

def BCE_with_sti(hard_sample_mining=0, num_back=None):
    def BCE_with_sample_type_indicator(y_true, y_pred):
        return reduce_sum(y_true[:, 0] * binary_crossentropy(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]))
    def BCE_with_sti_and_hsm(y_true, y_pred):
        return reduce_sum(tf.sort(y_true[:, 0] * binary_crossentropy(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]), direction='DESCENDING')[:num_back])
    if hard_sample_mining == 1:
        return BCE_with_sti_and_hsm
    elif hard_sample_mining == 0:
        return BCE_with_sample_type_indicator
    else:
        raise ValueError('hard_sample_mining must be 0 or 1, but got ' + str(hard_sample_mining) + ' instead.')
        
def MSE_with_sti(hard_sample_mining=0, num_back=None):
    def MSE_with_sample_type_indicator(y_true, y_pred):
        return reduce_sum(y_true[:, 0] * MSE(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]))
    def MSE_with_sti_and_hsm(y_true, y_pred):
        return reduce_sum(tf.sort(y_true[:, 0] * MSE(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]), direction='DESCENDING')[:num_back])
    if hard_sample_mining == 1:
        return MSE_with_sti_and_hsm
    elif hard_sample_mining == 0:
        return MSE_with_sample_type_indicator
    else:
        raise ValueError('hard_sample_mining must be 0 or 1, but got ' + str(hard_sample_mining) + ' instead.')