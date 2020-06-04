import numpy as np

def calibrate(bboxes):
    """
    This function will calibrate bounding boxes using the predicted bounding box.
    Args:
    - bboxes: ndarray of shape (n, 4) containing n bounding boxes. Each bounding
    box is a vector with the structure (x, y, width, height) where x, y is the
    coordinates of the top left point. The values of the coordinates are in the
    range [0, 1].

    Return:
    ndarray of shape (n, 3) containing n calibrated bounding boxes.
    """
    # Fix some bad coordinates
    calibrated_bboxes = np.clip(bboxes, 0, 1)
    bad_x_indices = calibrated_bboxes[:, 0] + calibrated_bboxes[:, 2] > 1
    calibrated_bboxes[bad_x_indices][:, 2] = 1 - calibrated_bboxes[bad_x_indices][:, 0]
    bad_y_indices = calibrated_bboxes[:, 1] + calibrated_bboxes[:, 3] > 1
    calibrated_bboxes[bad_y_indices][:, 3] = 1 - calibrated_bboxes[bad_y_indices][:, 1]
    # Calibrate
    size = np.max(calibrated_bboxes[:, 2:], axis=1, keepdims=True)
    delta = size - calibrated_bboxes[:, 2:]
    calibrated_bboxes[:, :2] = calibrated_bboxes[:, :2] - delta / 2
    calibrated_bboxes[:, 2] = size[:, 0]
    return calibrated_bboxes[:, :3]