import numpy as np
import cv2

def find_iou(window, bboxes):
    """
    Return the Intersection over Union
    Args:
    - window: a squared window with the format [x, y, size]
    - bboxes: ndarray of shape (n, 3) or (n, 4) containing n squared or
    rectangular bounding boxes
    Return: ndarray of shape (n,) 
    """
    lowest = np.minimum(bboxes[:, :2], window[:2])
    highest = np.maximum(bboxes[:, :2] + bboxes[:, 2:], window[:2] + window[2])
    intersection = window[2] + bboxes[:, 2:] - highest + lowest
    intersection = np.clip(intersection, 0, None)
    intersection = np.prod(intersection, axis=1)
    if bboxes.shape[1] == 3:
        union = bboxes[:, 2] ** 2 + window[2] ** 2 - intersection
    elif bboxes.shape[1] == 4:
        union = bboxes[:, 2] * bboxes[:, 3] + window[2] ** 2 - intersection
    else:
        raise ValueError('bboxes.shape[1] must be 3 or 4, but got ' + str(bboxes.shape[1]) + ' instead.')
    if np.count_nonzero(union == 0) > 0:
        print('\nfail')
        print(window)
        print(bboxes[union == 0])
        print(intersection[union == 0])
        exit()
    return intersection / union

def find_iou_bulk(windows, bboxes):
    """
    Like find_iou but works for multiple windows
    """
    lowest = np.minimum(windows[:, :2].reshape((-1, 1, 2)), bboxes[:, :2].reshape((1, -1, 2)))
    highest = np.maximum(windows[:, :2].reshape((-1, 1, 2)) + windows[:, 2].reshape((-1, 1, 1)), bboxes[:, :2].reshape((1, -1, 2)) + bboxes[:, 2:].reshape((1, -1, 2)))
    intersection = windows[:, 2].reshape((-1, 1, 1)) + bboxes[:, 2:].reshape((1, -1, 2)) - highest + lowest
    intersection = np.clip(intersection, 0, None)
    intersection = np.prod(intersection, axis=2)
    union = windows[:, 2].reshape((-1, 1)) ** 2 + bboxes[:, 2].reshape((1, -1)) * bboxes[:, 3].reshape((1, -1)) - intersection
    return intersection / union

def create_bbr_annotation(window, ground_truth):
    """
    Create bounding box regression annotation
    Args:
    - window: (3,) ndarray
    - ground_truth: (4,) ndarray
    Return: (4,) ndarray [x, y, width, height]
    """

    # Find x
    if window[0] <= ground_truth[0]:
        x = ground_truth[0] - window[0]
    else:
        x = 0
    
    # Find y
    if window[1] <= ground_truth[1]:
        y = ground_truth[1] - window[1]
    else:
        y = 0

    # Find width
    if window[0] + window[2] <= ground_truth[0] + ground_truth[2]:
        width = window[2] - x
    else:
        width = ground_truth[0] + ground_truth[2] - window[0] - x

    # Find height
    if window[1] + window[2] <= ground_truth[1] + ground_truth[3]:
        height = window[2] - y
    else:
        height = ground_truth[1] + ground_truth[3] - window[1] - y

    # Return result
    return np.divide(np.asarray([x, y, width, height]), window[2], dtype=np.float32)

def crop_and_resize(img, window, size):
    """
    Crop out the window and resize it to "size x size"
    Args:
    - img: the image
    - window: the window
    - size: 12, 24 or 48
    Return: The cropped and resized image
    """

    # Crop the image
    cropped = img[window[1]:window[1] + window[2], window[0]:window[0] + window[2]]

    # Resize if necessary
    if window[2] != size:
        return cv2.resize(cropped, (size, size))
    
    # Return cropped image
    return cropped