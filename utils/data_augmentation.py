import tensorflow as tf
    
def augment(images, labels):
    """
    Data augmentation for training MTCNN. This function will take in
    a batch of images and randomly flip, adjust brightness, contrast,
    hue, saturation of those images.
    
    Args:
    - images: a 4-D tensor of shape [batch size, height, width, channels]
    containing a batch of images.
    - labels: a tuple of 3 tensors of shape [batch size, 3],
    [batch size, 5], [batch size, 11] which are the labels for
    face classification, bounding box regression and facial landmarks
    localization.

    Return:
    a tuple of new (images, labels) that has been augmented
    """

    # Get the images, bounding boxes and landmarks from the input batch
    bboxes = labels[1]
    landmarks = labels[2]
    batch_size = images.shape[0]

    # Randomly flip the images horizontaly
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.flip_left_right(tf.gather_nd(images, random_indices)))
    
    # Randomly flip the bounding boxes
    random_bb = tf.gather_nd(bboxes, random_indices)
    flipped_bb_x = (1 - random_bb[:, 1] - random_bb[:, 3]) * random_bb[:, 0]
    flipped_bb = tf.concat([tf.reshape(random_bb[:, 0], [-1, 1]), tf.reshape(flipped_bb_x, [-1, 1]), random_bb[:, 2:]], axis=1)
    bboxes = tf.tensor_scatter_nd_update(bboxes, random_indices, flipped_bb)
    
    # Randomly flip the landmarks
    random_lm = tf.gather_nd(landmarks, random_indices)
    flipped_lm_xs = (1 - random_lm[:, 1::2]) * tf.reshape(random_lm[:, 0], [-1, 1])
    flipped_lm = tf.concat(
        values=[
            tf.reshape(random_lm[:, 0], [-1, 1]), # sample type indicator
            tf.reshape(flipped_lm_xs[:, 1], [-1, 1]), # left eye x
            tf.reshape(random_lm[:, 4], [-1, 1]), # left eye y
            tf.reshape(flipped_lm_xs[:, 0], [-1, 1]), # right eye x
            tf.reshape(random_lm[:, 2], [-1, 1]), # right eye y
            tf.reshape(flipped_lm_xs[:, 2], [-1, 1]), # nose x
            tf.reshape(random_lm[:, 6], [-1, 1]), # nose y
            tf.reshape(flipped_lm_xs[:, 4], [-1, 1]), # left mouth x
            tf.reshape(random_lm[:, 10], [-1, 1]), # left mouth y
            tf.reshape(flipped_lm_xs[:, 3], [-1, 1]), # right mouth x
            tf.reshape(random_lm[:, 8], [-1, 1]) # right mouth y
        ],
        axis=1
    )
    landmarks = tf.tensor_scatter_nd_update(landmarks, random_indices, flipped_lm)

    # Randomly adjust the hue
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.random_hue(tf.gather_nd(images, random_indices), 0.02))

    # Randomly adjust the saturation
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.random_saturation(tf.gather_nd(images, random_indices), 0.95, 1.05))

    # Randomly adjust the contrast
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.random_contrast(tf.gather_nd(images, random_indices), 0.95, 1.05))

    # Randomly adjust the brightness
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.random_brightness(tf.gather_nd(images, random_indices), 0.05))

    # Normalize the images
    images = (tf.cast(images, tf.float32) - 127.5) / 127.5

    return (images, (labels[0], bboxes, landmarks))

def augment_v2(images, labels):
    """
    Data augmentation for training MTCNN. This function will take in
    a batch of images and randomly flip, adjust brightness, contrast,
    hue, saturation of those images.
    
    Args:
    - images: a 4-D tensor of shape [batch size, height, width, channels]
    containing a batch of images.
    - labels: a tuple of 3 tensors of shape [batch size, 3],
    [batch size, 5], [batch size, 11] which are the labels for
    face classification, bounding box regression and facial landmarks
    localization.

    Return:
    a tuple of new (images, labels) that has been augmented
    """

    # Get the images, bounding boxes and landmarks from the input batch
    bboxes = labels[1]
    landmarks = labels[2]
    batch_size = images.shape[0]

    # Randomly flip the images horizontaly
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.flip_left_right(tf.gather_nd(images, random_indices)))
    
    # Randomly flip the bounding boxes
    random_bb = tf.gather_nd(bboxes, random_indices)
    flipped_bb_x1 = (1 - random_bb[:, 4]) * random_bb[:, 0]
    flipped_bb_x2 = (1 - random_bb[:, 2]) * random_bb[:, 0]
    flipped_bb = tf.concat([random_bb[:, :2], tf.reshape(flipped_bb_x1, [-1, 1]), tf.reshape(random_bb[:, 3], [-1, 1]), tf.reshape(flipped_bb_x2, [-1, 1])], axis=1)
    bboxes = tf.tensor_scatter_nd_update(bboxes, random_indices, flipped_bb)
    
    # Randomly flip the landmarks
    random_lm = tf.gather_nd(landmarks, random_indices)
    flipped_lm_xs = (1 - random_lm[:, 2::2]) * tf.reshape(random_lm[:, 0], [-1, 1])
    flipped_lm = tf.concat(
        values=[
            tf.reshape(random_lm[:, 0], [-1, 1]), # sample type indicator
            tf.reshape(random_lm[:, 3], [-1, 1]), # left eye y
            tf.reshape(flipped_lm_xs[:, 1], [-1, 1]), # left eye x
            tf.reshape(random_lm[:, 1], [-1, 1]), # right eye y
            tf.reshape(flipped_lm_xs[:, 0], [-1, 1]), # right eye x
            tf.reshape(random_lm[:, 5], [-1, 1]), # nose y
            tf.reshape(flipped_lm_xs[:, 2], [-1, 1]), # nose x
            tf.reshape(random_lm[:, 9], [-1, 1]), # left mouth y
            tf.reshape(flipped_lm_xs[:, 4], [-1, 1]), # left mouth x
            tf.reshape(random_lm[:, 7], [-1, 1]), # right mouth y
            tf.reshape(flipped_lm_xs[:, 3], [-1, 1]), # right mouth x
        ],
        axis=1
    )
    landmarks = tf.tensor_scatter_nd_update(landmarks, random_indices, flipped_lm)

    # Randomly adjust the hue
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.random_hue(tf.gather_nd(images, random_indices), 0.02))

    # Randomly adjust the saturation
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.random_saturation(tf.gather_nd(images, random_indices), 0.95, 1.05))

    # Randomly adjust the contrast
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.random_contrast(tf.gather_nd(images, random_indices), 0.95, 1.05))

    # Randomly adjust the brightness
    random_indices = tf.where(tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32))
    images = tf.tensor_scatter_nd_update(images, random_indices, tf.image.random_brightness(tf.gather_nd(images, random_indices), 0.05))

    # Normalize the images
    images = (tf.cast(images, tf.float32) - 127.5) / 127.5

    return (images, (labels[0], bboxes, landmarks))