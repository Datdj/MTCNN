import tensorflow as tf
from tensorflow.math import reduce_sum

def accuracy_(threshold=0.5):
    """
    Accuracy metric for MTCNN. This function will predict whether an image has
    a face or not using the first output of the model. If the model output a
    probability >= a threshold then that image is predicted as having a face.
    After that, it will compare the result with the ground truth to compute 
    accuracy. If a batch has no positives and negatives, the accurary will be
    defaulted to 0.5 (approximated random chance).

    Args:
    - threshold: a float threshold in range [0, 1]. Default is 0.5

    Return:
    - a function with the signature result = fn(y_true, y_pred)
    """
    def accuracy(y_true, y_pred):
        def return_0_5():
            return 0.5
        def return_acc():
            return count / total
        total = reduce_sum(y_true[:, 0])
        pred = tf.cast(y_pred[:, 1] >= threshold, dtype=tf.float32) * y_true[:, 0]
        matches = tf.cast(pred == y_true[:, 1], dtype=tf.float32) * y_true[:, 0]
        count = reduce_sum(matches)
        return tf.cond(total == 0, return_0_5, return_acc)
    return accuracy

def recall_(threshold=0.5):
    """
    Recall metric for MTCNN. This function will predict whether an image has
    a face or not using the first output of the model. If the model output a
    probability >= a threshold then that image is predicted as having a face.
    After that, it will compare the result with the ground truth to compute 
    recall. If a batch has no positives and negatives, the recall will be
    defaulted to 0.5 (approximated random chance).

    Args:
    - threshold: a float threshold in range [0, 1]. Default is 0.5

    Return:
    - a function with the signature result = fn(y_true, y_pred)
    """
    def recall(y_true, y_pred):
        def return_0_5():
            return 0.5
        def return_rec():
            return count / total
        total = reduce_sum(y_true[:, 1])
        pred = tf.cast(y_pred[:, 1] >= threshold, dtype=tf.float32) * y_true[:, 0]
        count = reduce_sum(tf.cast(y_true[:, 1] + pred == 2, dtype=tf.float32))
        return tf.cond(total == 0, return_0_5, return_rec)
    return recall