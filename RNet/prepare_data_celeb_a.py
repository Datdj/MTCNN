import sys
sys.path.insert(0, '..')
import argparse
import cv2
import numpy as np
from os import path
import os
from tqdm import tqdm
from numpy.random import default_rng
from PNet.model import pnet_predict
import tensorflow as tf
from utils.misc import check_landmarks, find_iou_bulk, crop_and_resize_v2
from matplotlib import pyplot as plt
import matplotlib
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    parser.add_argument(
        '--model-path',
        type=str,
        help='The path to the .hdf5 file containing the weights of P-Net',
        required=True
    )
    args = parser.parse_args()

    rng = default_rng()

    # Prepare saving folder
    if not path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Load P-Net
    pnet = pnet_predict()
    pnet.load_weights(filepath=args.model_path, by_name=True)

    # Use this to save images as numpy file
    lm_images = list()

    # Open the annotation files
    bb = open(args.bounding_boxes)
    lm = open(args.landmarks)

    # Get the number of faces
    num_faces = int(lm.readline())
    
    # Skip some unnecessary lines
    bb.readline()
    bb.readline()
    lm.readline()

    # Start creating landmark faces
    norm_landmarks = list()
    for i in tqdm(range(num_faces)):

        # Read the annotation files line by line
        bb_line = bb.readline().split()
        bbox = np.array(bb_line[1:], dtype=np.int32)
        # print(bbox)
        bbox[:2] = bbox[:2][::-1]
        bbox[2:] = bbox[2:][::-1]
        # print(bbox)
        # exit()
        lm_line = lm.readline().split()
        landmarks = np.reshape(np.array(lm_line[1:], dtype=np.int32), (5, 2))[:, ::-1]
        # print(landmarks)
        # exit()

        # if rng.random() > 0.0001:
        #     continue

        # Check for images that have wrong bounding box annotation and skip them
        if np.count_nonzero(np.min(landmarks, 0) < bbox[:2]) != 0 or np.count_nonzero(np.max(landmarks, 0) > bbox[:2] + bbox[2:] - 1) != 0:
            continue

        # Read the image
        img = cv2.imread(args.images_directory + '/' + lm_line[0])[:, :, ::-1]
        # print(img.shape)
        # exit()

        # Get the original image size
        height, width = img.shape[:2]
        
        min_size = np.min(bbox[2:])
        # print(type(min_size), type(min_size * 0.7))
        min_face = int(min_size * 0.7)
        # print(type(min_face))
        # print(bbox)
        # print(min_size)
        # print(min_face)
        # exit()

        # Initialize the scale
        scale = 12 / min_face
        # print(type(scale))
        # exit()
        h = round(height * scale)
        w = round(width * scale)

        # print(type(h), type(w))
        # exit()
        c_bboxes = list()
        confidence = list()

        # Start the loop
        while h >= 12 and w >= 12:
            
            # Resize the image
            resized_img = cv2.resize(img, (w, h)).reshape((1, h, w, 3))

            # Preprocess the image
            resized_img = np.add(resized_img, -127.5, dtype=np.float32) / 127.5

            # Pass the image to P-Net
            c_bboxes_i, confidence_i = pnet(resized_img)
            
            # If there are no faces, move on to next loop
            if c_bboxes_i.shape[0] == 0:
                scale = scale * 0.7
                h = round(height * scale)
                w = round(width * scale)
                continue

            c_bboxes.append(c_bboxes_i / scale)
            confidence.append(confidence_i)

            # Calculate the next scale
            scale = scale * 0.7
            h = round(height * scale)
            w = round(width * scale)

        # Apply non maximum suppression one last time
        if len(c_bboxes) == 0:
            continue
        c_bboxes = tf.concat(c_bboxes, axis=0)
        if len(confidence) > 1:
            confidence = tf.concat(confidence, axis=0)
            final_indices = tf.image.non_max_suppression(c_bboxes, confidence, confidence.shape[0], 0.7)
            c_bboxes = tf.gather(c_bboxes, final_indices)
        # print(c_bboxes.shape)

        # Round out the values
        c_bboxes = tf.cast(tf.round(c_bboxes), tf.int32)

        c_bboxes = c_bboxes.numpy()[:, :3]
        c_bboxes[:, 2] = c_bboxes[:, 2] - c_bboxes[:, 0]
        # print(c_bboxes)
        # exit()

        # Get the window that has all the landmarks
        windows = check_landmarks(c_bboxes, landmarks)
        if windows.size == 0:
            continue
        # print(windows)
        ious = find_iou_bulk(windows, bbox.reshape((1, 4)))[:, 0] 
        # print(ious)
        # show(img, c_bboxes=windows, bbox_=bbox)
        # exit()

        # Only keep the windows that has iou > 0.4
        windows = windows[ious > 0.4]
        windows = windows[windows[:, 2] * windows[:, 2] < 1.2 * bbox[2] * bbox[3]]
    #     print(windows[:, 2] * windows[:, 2])
    #     print(bbox[2] * bbox[3])
    #     print(windows)
    #     print(bbox)
    #     print(windows[:, 2] * windows[:, 2] < 1.2 * bbox[2] * bbox[3])
        # show(img, c_bboxes=windows, bbox_=bbox)
    #     # exit()
    #     continue
    # exit()
        # print(windows)
        # print(landmarks)
        # exit()

        if windows.size == 0:
            continue

        for window in windows:

            # Crop out the landmark faces and resize them to 24 x 24            
            lm_images.append(crop_and_resize_v2(img, window, 24))

            # Normalize the landmarks coordinates
            norm_landmarks.append(np.divide(landmarks - window[:2], window[2], dtype=np.float32).reshape(10))
        
        # print(norm_landmarks)
        # exit()

    # Close the annotation files
    bb.close()
    lm.close()

    # Save the images
    np.save(args.output_directory + '/lm_images.npy', np.asarray(lm_images))

    # Save the normalized landmarks into a numpy file
    np.save(args.output_directory + '/landmarks.npy', np.asarray(norm_landmarks))

def show(img, c_bboxes, bbox_):
    h, w = img.shape[:2]
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = w / dpi, h / dpi
    figure, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    for bbox in c_bboxes:
        bb = matplotlib.patches.Rectangle(xy=(bbox[1], bbox[0]), width=bbox[2], height=bbox[2], fill=False, edgecolor='g')
        ax.add_patch(bb)
    bb = matplotlib.patches.Rectangle(xy=(bbox_[1], bbox_[0]), width=bbox_[3], height=bbox_[2], fill=False, edgecolor='r')
    ax.add_patch(bb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()