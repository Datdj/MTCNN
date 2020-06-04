import numpy as np

def calibrate(bboxes):
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