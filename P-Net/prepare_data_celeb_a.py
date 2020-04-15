import argparse
import cv2
import numpy as np
import pandas as pd
from os import path
import os
from tqdm import tqdm

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
        required=True
    )
    args = parser.parse_args()

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
        os.mkdir(args.output_directory)
    if not path.exists(args.output_directory + '/landmarks'):
        os.mkdir(args.output_directory + '/landmarks')

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
            window[1] = helper(y1, y2, nose_y, delta, window[2])
        elif bbox[2] > bbox[3]:
            delta = bbox[2] - bbox[3]
            window[1] = bbox[1]
            window[2] = bbox[3]
            nose_x = landmarks[2, 0]
            x1 = bbox[0]
            x2 = x1 + bbox[2] - 1
            window[0] = helper(x1, x2, nose_x, delta, window[2])
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
        cv2.imwrite(
            filename=args.output_directory + '/landmarks/' + bb_line[0],
            img=resized_window
        )

        # Normalize the landmarks coordinates
        norm_landmarks = np.vstack((norm_landmarks, ((landmarks - window[:2]) / (window[2] - 1)).reshape(10)))

    # Close the annotation files
    bb.close()
    lm.close()

    # Save the normalized landmarks into an excel file
    pd.DataFrame(norm_landmarks).to_excel(args.output_directory + '/landmarks.xlsx')

# An attempt at making the codes above look like less of a mess :v
def helper(x1, x2, nose, delta, crop_size):
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
    return max(0, result + ((-0.05 + np.random.random() / 10) * crop_size).astype(np.int32))

if __name__ == "__main__":
    main()