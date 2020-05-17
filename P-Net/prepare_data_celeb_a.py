import argparse
import cv2
import numpy as np
import pandas as pd
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

    # # Show some images with their respective bounding boxes
    # f = open(args.bounding_boxes)
    # f.readline()
    # f.readline()
    # for i in range(50):
    #     test = f.readline()
    #     name = test.split()[0]
    #     bbox = np.array(test.split()[1:], np.int32)
    #     img = cv2.imread(args.images_directory + '/' + name)
    #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] -1), (0, 255, 0))
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # f.close()

    # Prepare saving folder
    if not path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    if not path.exists(args.output_directory + '/landmarks'):
        os.mkdir(args.output_directory + '/landmarks')

    # Use this to save images as numpy file
    lm_images = np.zeros((0, 12, 12, 3), np.uint8)

    # Save the original windows
    lm_origin_windows = np.zeros((0, 3), np.int32)
    lm_origin_names = []

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
    norm_landmarks = np.zeros((0, 10))
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
        img = cv2.imread(args.images_directory + '/' + bb_line[0])

        # Resize the window to 12 x 12
        resized_window = cv2.resize(
            src=img[window[1]:window[1] + window[2], window[0]:window[0] + window[2], :],
            dsize=(12, 12),
            interpolation=cv2.INTER_AREA
        )

        # Save the image
        lm_images = np.concatenate((lm_images, resized_window.reshape((1, 12, 12, 3))), 0)
        lm_origin_windows = np.concatenate((lm_origin_windows, window.reshape((1, 3))), axis=0)
        lm_origin_names.append(bb_line[0])
        cv2.imwrite(
            filename=args.output_directory + '/landmarks/' + bb_line[0],
            img=resized_window
        )

        # Normalize the landmarks coordinates
        norm_landmarks = np.vstack((norm_landmarks, ((landmarks - window[:2]) / (window[2] - 1)).reshape(10)))

    # Close the annotation files
    bb.close()
    lm.close()

    # Save the images
    np.save(args.output_directory + '/lm_images.npy', lm_images)

    # Save the original windows and file names
    pd.DataFrame(
        data={
            'file_name': lm_origin_names,
            'x': lm_origin_windows[:, 0],
            'y': lm_origin_windows[:, 1],
            'window_size': lm_origin_windows[:, 2]
        }
    ).to_excel(args.output_directory + '/lm_origin.xlsx')

    # Save the normalized landmarks into an excel and numpy file
    np.save(args.output_directory + '/landmarks.npy', norm_landmarks)
    pd.DataFrame(norm_landmarks).to_excel(args.output_directory + '/landmarks.xlsx')

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