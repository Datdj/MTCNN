import numpy as np

def get_windows_vectorized(img, stride=1):
    h, w = img.shape[:2]
    h = int((h - 12) / stride) + 1
    w = int((w - 12) / stride) + 1
    return np.lib.stride_tricks.as_strided(img, shape=(h, w, 12, 12, 3), strides=(stride * img.strides[0], stride * img.strides[1], img.strides[0], img.strides[1], img.strides[2]), writeable=False)