from tensorflow.math import reduce_sum
from tensorflow.keras.losses import binary_crossentropy, MSE

def BCE_with_sti(hard_sample_mining=False, num_back=None):
    def BCE_with_sample_type_indicator(y_true, y_pred):
        return reduce_sum(y_true[:, 0] * binary_crossentropy(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]))
    def BCE_with_sti_and_hsm(y_true, y_pred):
        return reduce_sum(tf.sort(y_true[:, 0] * binary_crossentropy(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]), direction='DESCENDING')[:num_back])
    if hard_sample_mining == True:
        return BCE_with_sti_and_hsm
    else:
        return BCE_with_sample_type_indicator
        
def MSE_with_sti(hard_sample_mining=False, num_back=None):
    def MSE_with_sample_type_indicator(y_true, y_pred):
        return reduce_sum(y_true[:, 0] * MSE(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]))
    def MSE_with_sti_and_hsm(y_true, y_pred):
        return reduce_sum(tf.sort(y_true[:, 0] * MSE(y_true=y_true[:, 1:], y_pred=y_pred[:, 1:]), direction='DESCENDING')[:num_back])
    if hard_sample_mining == True:
        return MSE_with_sti_and_hsm
    else:
        return MSE_with_sample_type_indicator