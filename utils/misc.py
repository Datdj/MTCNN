import numpy as np
import cv2
# import time

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

def create_bbr_annotation_v2(window, ground_truth):
    """
    Create bounding box regression annotation
    Args:
    - window: (3,) ndarray
    - ground_truth: (4,) ndarray
    Return: (4,) ndarray [y1, x1, y2, x2]
    """

    # print(window)
    # print(ground_truth)
    y1_x1 = np.maximum(window[:2], ground_truth[:2]) - window[:2]
    y2_x2 = np.minimum(window[:2] + window[2], ground_truth[:2] + ground_truth[2:]) - window[:2]
    return np.divide(np.concatenate((y1_x1, y2_x2)), window[2], dtype=np.float32)

def create_bbr_annotation_v2_bulk(windows, ground_truth):
    """
    Create bounding box regression annotations
    Args:
    - windows: 2d array (n, 3)
    - ground_truth: 2d array (n, 4)
    Return: 2d array (n, 4)
    """

    y1_x1 = np.maximum(windows[:, :2], ground_truth[:, :2]) - windows[:, :2]
    y2_x2 = np.minimum(windows[:, :2] + windows[:, 2:], ground_truth[:, :2] + ground_truth[:, 2:]) - windows[:, :2]
    return np.divide(np.concatenate((y1_x1, y2_x2), axis=1), windows[:, 2:], dtype=np.float32)

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

def crop_and_resize_v2(img, window, size):
    """
    Crop out the window and resize it to "size x size"
    Args:
    - img: the image
    - window: the window [y, x, window_size]
    - size: 12, 24 or 48
    Return: The cropped and resized image
    """

    pad_size = 0
    # test = False

    if window[0] < 0:
        pad_size = max(-window[0], pad_size)

    if window[1] < 0:
        pad_size = max(-window[1], pad_size)
        # print(window, img.shape)

        # assert False

    if window[0] + window[2] > img.shape[0]:
        pad_size = max(window[0] + window[2] - img.shape[0], pad_size)
        # test = True
        # print(window, img.shape)
        # assert False

    if window[1] + window[2] > img.shape[1]:
        pad_size = max(window[1] + window[2] - img.shape[1], pad_size)
        # test = True
        # print(window, img.shape)
        # assert False

    # Pad the image
    if pad_size > 0:
        # if test == True:
        #     print(window, img.shape)
        img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))
        window[:2] = window[:2] + pad_size
        # if test == True:
        #     print(window, img.shape)
        #     assert False

    # Crop the image
    cropped = img[window[0]:window[0] + window[2], window[1]:window[1] + window[2]]

    # Resize if necessary
    if window[2] != size:
        return cv2.resize(cropped, (size, size))
    
    # Return cropped image
    return cropped

def check_landmarks(windows, landmarks):
    """
    Return windows that have all the landmarks inside.
    Args:
    - windows: (n, 3) ndarray
    - landmarks: (5, 2) ndarray
    Return:
    - lm_windows: (m, 3)
    """
    
    # The landmarks have to be inside windows and at least a margin away from the edges of that window 
    margin = (windows[:, 2] // 10)
    # print(type(margin))
    # print(margin.shape)
    # print(margin.dtype)
    # print(margin)

    # windows_ is the smaller windows when we cut the margin off
    windows_ = np.zeros_like(windows)
    windows_[:, :2] = windows[:, :2] + margin.reshape((-1, 1))
    windows_[:, 2] = windows[:, 2] - 2 * margin
    # print(windows, windows_)

    # tic = time.time()
    result = []
    for i, window_ in enumerate(windows_):
        if np.count_nonzero(np.min(landmarks, 0) < window_[:2]) == 0 and np.count_nonzero(np.max(landmarks, 0) > window_[:2] + window_[2] - 1) == 0:
            result.append(windows[i])
            # print(window_)
            # print(windows[i])
    # toc = time.time()
    # print(toc - tic)

    return np.asarray(result)