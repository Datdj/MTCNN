import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib

def show(batch, mean_image=None, bgr=False):
    """
    Visualize a batch of data in MTCNN model.

    Args:
    - batch: an element in the dataset.
    - mean_image: the value used to zero center the images
    - bgr: True if the input is bgr images
    """
    (x, (y1, y2, y3)) = batch
    if mean_image is not None:
        x = x + mean_image
    for i in range(x.shape[0]):
        if bgr == True:
            rgb_img = tf.stack([x[i, :, :, 2], x[i, :, :, 1], x[i, :, :, 0]], axis=2)
        else:
            rgb_img = x[i]
        figure, ax = plt.subplots(1)
        ax.imshow(rgb_img, extent=(0, 12, 12, 0))
        if y1[i, 0] == 1 and y2[i, 0] == 1:
            sample_type = 'positive'
            bb = matplotlib.patches.Rectangle((y2[i, 1] * 12, y2[i, 2] * 12), width=y2[i, 3] * 12, height=y2[i, 4] * 12, fill=False, edgecolor='g')
            ax.add_patch(bb)
        elif y1[i, 0] == 0 and y2[i, 0] == 1:
            sample_type = 'part face'
            bb = matplotlib.patches.Rectangle((y2[i, 1] * 12, y2[i, 2] * 12), width=y2[i, 3] * 12, height=y2[i, 4] * 12, fill=False, edgecolor='g')
            ax.add_patch(bb)
        elif y1[i, 0] == 1 and y1[i, 1] == 0:
            sample_type = 'negative'
        elif y3[i, 0] == 1:
            sample_type = 'landmark'
            x_coord = (y3[i, 1::2] * 12).numpy()
            y_coord = (y3[i, 2::2] * 12).numpy()
            plt.scatter(x_coord, y_coord)
        plt.title(sample_type)
        ax.axis('off')