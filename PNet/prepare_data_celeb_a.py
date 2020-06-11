import argparse
import cv2
import numpy as np
from os import path
import os
from tqdm import tqdm
from numpy.random import default_rng

def main():

    parser = argparse.ArgumentParser(description='Create landmark faces for P-Net using CelebA dataset')
    parser.add_argument(
        '-b',
        '--bounding-boxes',
        type=str,
        help='The file that contains bounding boxes for CelebA dataset',
        required=True
    )
    parser.add_argument(
        '-l',
        '--landmarks',
        type=str,
        help='The file that contains landmark annotations of CelebA dataset',
        required=True
    )
    parser.add_argument(
        '-i',
        '--images-directory',
        type=str,
        help='The path to the folder that contains CelebA dataset',
        required=True
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        type=str,
        help='The path to the folder where you want to save your output',
        default='./data/temp'
    )
    args = parser.parse_args()

    rng = default_rng()

    # Prepare saving folder
    if not path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Use this to save images as numpy file
    lm_images = list()

    # Open the annotation files
    bb = open(args.bounding_boxes)
    lm = open(args.landmarks)

    # Get the number of faces
    num_faces = int(bb.readline())
    
    # Skip some unnecessary lines
    bb.readline()
    lm.readline()
    lm.readline()

    # Start creating landmark faces
    norm_landmarks = list()
    for i in tqdm(range(num_faces)):

        # Read the annotation files line by line
        bb_line = bb.readline().split()
        bbox = np.array(bb_line[1:], dtype=np.int32)
        lm_line = lm.readline().split()
        landmarks = np.reshape(np.array(lm_line[1:], dtype=np.int32), (5, 2))

        # Check for images that have wrong bounding box annotation and skip them
        if np.count_nonzero(np.min(landmarks, 0) < bbox[:2]) != 0 or np.count_nonzero(np.max(landmarks, 0) > bbox[:2] + bbox[2:] - 1) != 0:
            continue

        # The crazy part where no amount of documentation can make it easier to read =((
        # But basically these codes are used to calculate the window that are going to be cropped out of the image
        window = np.zeros(3, dtype=np.int32)
        if bbox[2] < bbox[3]:
            delta = bbox[3] - bbox[2]
            window[0] = bbox[0]
            window[2] = bbox[2]
            nose_y = landmarks[2, 1]
            y1 = bbox[1]
            y2 = bbox[1] + bbox[3] - 1
            window[1] = helper(y1, y2, nose_y, delta, window[2], rng)
        elif bbox[2] > bbox[3]:
            delta = bbox[2] - bbox[3]
            window[1] = bbox[1]
            window[2] = bbox[3]
            nose_x = landmarks[2, 0]
            x1 = bbox[0]
            x2 = x1 + bbox[2] - 1
            window[0] = helper(x1, x2, nose_x, delta, window[2], rng)
        else:
            window = bbox[:3]

        # Double check to see if there are any landmarks outside the window
        if np.count_nonzero(np.min(landmarks, 0) < window[:2]) != 0 or np.count_nonzero(np.max(landmarks, 0) > window[:2] + window[2] - 1) != 0:
            continue
        
        # Read the image
        img = cv2.imread(args.images_directory + '/' + bb_line[0])[:, :, ::-1]

        # Resize the window to 12 x 12
        resized_window = cv2.resize(
            src=img[window[1]:window[1] + window[2], window[0]:window[0] + window[2], :],
            dsize=(12, 12)
        )

        lm_images.append(resized_window)

        # Normalize the landmarks coordinates
        norm_landmarks.append(np.divide(landmarks - window[:2], window[2], dtype=np.float32).reshape(10))

    # Close the annotation files
    bb.close()
    lm.close()

    # Save the images
    np.save(args.output_directory + '/lm_images.npy', np.asarray(lm_images))

    # Save the normalized landmarks into a numpy file
    np.save(args.output_directory + '/landmarks.npy', np.asarray(norm_landmarks))

# An attempt at making the codes above look like less of a mess :v
def helper(x1, x2, nose, delta, crop_size, rng):
    left = nose - x1 + 1
    right = x2 - nose + 1
    if left > right:
        delta1 = left - right
        if delta1 < delta:
            delta2 = delta - delta1
            result = x1 + delta1 + (delta2 / 2).astype(np.int32)
        else:
            return x1 + delta
    elif left == right:
        result = x1 + (delta / 2).astype(np.int32)
    else:
        delta1 = right - left
        if delta1 < delta:
            delta2 = delta - delta1
            result = x1 + (delta2 / 2).astype(np.int32)
        else:
            return x1
    return max(0, result + ((-0.05 + rng.random() / 10) * crop_size).astype(np.int32))

if __name__ == "__main__":
    main()