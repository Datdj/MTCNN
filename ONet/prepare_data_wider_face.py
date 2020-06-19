import sys
sys.path.insert(0, '..')
import argparse
import cv2
import numpy as np
from os import path
import os
from numpy.random import default_rng
from tqdm import tqdm
from utils.misc import find_iou_bulk, create_bbr_annotation_v2_bulk, crop_and_resize_v2
from PNet.model import pnet_predict, pnet_predict_v2
from RNet.model import rnet_predict
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_arguments():
    """
    Get arguments from the command line.
    """
    parser = argparse.ArgumentParser(description='Create Positives, Negatives and Part Faces from the WIDER FACE dataset for R-Net.')
    parser.add_argument(
        '-a',
        '--annotation-file',
        type=str,
        help='The path to the annotation file of the WIDER FACE dataset',
        required=True
    )
    parser.add_argument(
        '-i',
        '--images-directory',
        type=str,
        help='The path to the folder that contains WIDER FACE dataset',
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
        '--pnet',
        type=str,
        help='The path to the .hdf5 file containing the weights of P-Net',
        required=True
    )
    parser.add_argument(
        '--rnet',
        type=str,
        help='The path to the .hdf5 file containing the weights of R-Net',
        required=True
    )
    return parser.parse_args()

def main():

    rng = default_rng()

    # Get arguments
    args = get_arguments()

    # Create saving folder
    if not path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Load P-Net
    # pnet = pnet_predict()
    pnet = pnet_predict_v2()
    pnet.load_weights(filepath=args.pnet, by_name=True)

    # Load R-Net
    rnet = rnet_predict()
    rnet.load_weights(filepath=args.rnet, by_name=True)

    # These are for storing the images
    pos_images = list()
    par_images = list()
    neg_images = list()

    # These are for storing the annotations of bounding box regression task
    pos_bboxes = list()
    par_bboxes = list()

    # Open the annotation file
    f = open(args.annotation_file)
    batch = 0
    for i in tqdm(range(12880)): # Because there are a total of 12880 images in the Wider Face dataset

        # Read the file line by line
        line = f.readline()

        # Obtain the number of faces in this image
        num_faces = int(f.readline())

        # Skip the image if it does not contain any faces
        if num_faces == 0:
            f.readline()
            continue
        
        # Obtain bounding boxes while taking into account invalid faces
        ground_truth = np.zeros((num_faces, 4), np.int32)
        invalid = []
        for j in range(num_faces):
            face = f.readline()
            if face[-7] == '1':
                invalid.append(j)
                continue
            ground_truth[j] = face[:-14].split(' ')

        # if rng.random() > 0.01:
        #     continue
        # count += 1

        # Remove invalid faces
        if len(invalid) > 0:
            ground_truth = np.delete(ground_truth, invalid, 0)
            if ground_truth.shape[0] == 0:
                continue
        # num_faces = ground_truth.shape[0]

        min_size = np.min(ground_truth[:, 2:])
        min_size = int(min_size * 0.6)
        min_size = max(min_size, 20)

        # Read the image
        img = cv2.imread(args.images_directory + '/' + line[:-1])[:, :, ::-1]
        # print(img.shape)

        # Get the original image size
        height, width = img.shape[:2]

        # Initialize the scale
        scale = 12 / min_size
        h = round(height * scale)
        w = round(width * scale)

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
                scale = scale * 0.8
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
        c_bboxes = tf.concat(c_bboxes, axis=0)
        # print(c_bboxes.shape)
        confidence = tf.concat(confidence, axis=0)
        final_indices = tf.image.non_max_suppression(c_bboxes, confidence, confidence.shape[0], 0.7)
        c_bboxes = tf.gather(c_bboxes, final_indices)

        c_bboxes = square(c_bboxes)

        # print(c_bboxes.shape)
        # exit()

        # Round out the values
        c_bboxes = tf.cast(tf.round(c_bboxes), tf.int32)
        # print(type(c_bboxes))
        # print(c_bboxes.shape)
        # print(c_bboxes[0])
        # exit()

        c_bboxes = c_bboxes.numpy()[:, :3]
        c_bboxes[:, 2] = c_bboxes[:, 2] - c_bboxes[:, 0]
        # print(type(c_bboxes))
        # print(c_bboxes.shape)
        # print(c_bboxes[0])
        # exit()

        # print(type(ground_truth))
        # print(ground_truth.shape)
        # print(ground_truth[0])
        # # exit()

        ground_truth[:, :2] = ground_truth[:, :2][:, ::-1]
        ground_truth[:, 2:] = ground_truth[:, 2:][:, ::-1]

        # temp = ground_truth[:, 0].copy()
        # ground_truth[:, 0] = ground_truth[:, 1]
        # ground_truth[:, 1] = temp
        # temp = ground_truth[:, 2].copy()
        # ground_truth[:, 2] = ground_truth[:, 3]
        # ground_truth[:, 3] = temp
        
        # print(type(ground_truth))
        # print(ground_truth.shape)
        # print(ground_truth[0])
        # exit()

        # print(ground_truth.shape)
        # print(c_bboxes.shape)

        

        # Create R-Net input
        rnet_input = list()
        for bbox in c_bboxes:
            # print(bbox)
            # exit()
            rnet_input.append(crop_and_resize_v2(img, bbox, 24))
        rnet_input = np.asarray(rnet_input)
        # inspect(rnet_input)

        # Preprocess R-Net input
        rnet_input = np.add(rnet_input, -127.5, dtype=np.float32) / 127.5

        # Stage 2
        # print(type(c_bboxes))
        # print(rnet_input.shape)
        # print(c_bboxes.shape)
        c_bboxes = rnet([rnet_input, c_bboxes])
        # print(c_bboxes)
        # print(type(c_bboxes))
        # print(c_bboxes.shape)
        # print(c_bboxes[0])
        # show(img, c_bboxes.numpy())
        # if count == 15:
        #     exit()
        # continue
        # exit()

        c_bboxes = c_bboxes.numpy()[:, :3]
        c_bboxes[:, 2] = c_bboxes[:, 2] - c_bboxes[:, 0]

        # Calculate iou
        ious = find_iou_bulk(c_bboxes, ground_truth)

        # print(c_bboxes.shape[0])
        # print(np.count_nonzero(ious > 0.65))
        # exit()

        # h, w = img.shape[:2]
        # dpi = matplotlib.rcParams['figure.dpi']
        # figsize = w / dpi, h / dpi
        # figure, ax = plt.subplots(1, figsize=figsize)
        # ax.imshow(img)
        # for bbox in c_bboxes:
        #     bb = matplotlib.patches.Rectangle(xy=(bbox[1], bbox[0]), width=bbox[2], height=bbox[2], fill=False, edgecolor='b')
        #     ax.add_patch(bb)

        # Start creating data
        num_windows = c_bboxes.shape[0]
        max_iou_mask = np.argmax(ious, axis=1)
        max_ious = ious[np.arange(num_windows), max_iou_mask]

        # print(max_ious.shape)
        # exit()

        # Create positives
        pos_mask = max_ious > 0.65
        num_pos = np.count_nonzero(pos_mask)
        if num_pos > 0:
            pos_windows = c_bboxes[pos_mask]
            # num_pos = pos_windows.shape[0]
            pos_bboxes_i = create_bbr_annotation_v2_bulk(pos_windows, ground_truth[max_iou_mask[pos_mask]])
            pos_bboxes.extend(pos_bboxes_i)
            for bbox in pos_windows:
                # print(bbox)
                # bb = matplotlib.patches.Rectangle(xy=(bbox[1], bbox[0]), width=bbox[2], height=bbox[2], fill=False, edgecolor='g')
                # ax.add_patch(bb)
                pos_images.append(crop_and_resize_v2(img, bbox, 48))
        # else:
        #     num_pos = 0
        # print(ground_truth[max_iou_mask[max_ious > 0.65]])
        # print(create_bbr_annotation_v2_bulk(c_bboxes[max_ious > 0.65], ground_truth[max_iou_mask[max_ious > 0.65]]))
        # print(pos_bboxes)
        # print(pos_images[0].shape)
        # print(pos_images[0].shape)
        # exit()

        # Create part faces
        par_mask = np.logical_and(0.45 <= max_ious, max_ious <= 0.65)
        num_par = np.count_nonzero(par_mask)
        if num_par > 0:
            par_windows = c_bboxes[par_mask]
            par_bboxes_i = create_bbr_annotation_v2_bulk(par_windows, ground_truth[max_iou_mask[par_mask]])
            par_bboxes.extend(par_bboxes_i)
            for bbox in par_windows:
                # bb = matplotlib.patches.Rectangle(xy=(bbox[1], bbox[0]), width=bbox[2], height=bbox[2], fill=False, edgecolor='y')
                # ax.add_patch(bb)
                par_images.append(crop_and_resize_v2(img, bbox, 24))
        #     print(par_windows)
        #     print(ground_truth)
        # print(np.count_nonzero(par_mask))
        # print(par_mask)
        # exit()

        
        
        

        # Create negatives
        # print(num_windows)
        neg_mask = max_ious < 0.3
        num_neg = np.count_nonzero(neg_mask)
        # print(num_neg)
        if num_neg > 0:
            count = 0
            limit = 100
            size = []
            indices = rng.permutation(np.arange(num_windows)[neg_mask])
            # print(indices)
            for j in indices:
                if c_bboxes[j, 2] not in size:
                    size.append(c_bboxes[j, 2])
                    # print(c_bboxes[j])
                    count = count + 1
                    neg_images.append(crop_and_resize_v2(img, c_bboxes[j], 48))
                    # bbox = c_bboxes[j]
                    # bb = matplotlib.patches.Rectangle(xy=(bbox[1], bbox[0]), width=bbox[2], height=bbox[2], fill=False, edgecolor='r')
                    # ax.add_patch(bb)
                elif rng.random() < 0.1:
                    count = count + 1
                    neg_images.append(crop_and_resize_v2(img, c_bboxes[j], 48))
                    # bbox = c_bboxes[j]
                    # bb = matplotlib.patches.Rectangle(xy=(bbox[1], bbox[0]), width=bbox[2], height=bbox[2], fill=False, edgecolor='r')
                    # ax.add_patch(bb)
                if count == limit:
                    break

        # plt.axis('off')
        # plt.show()
        # if i == 5:
        #     exit()

        
        if (i + 1) % 2000 == 0:

            # Save the images
            pos_images = np.asarray(pos_images)
            par_images = np.asarray(par_images)
            neg_images = np.asarray(neg_images)
            np.save(args.output_directory + '/positive_images_' + str(batch) + '.npy', pos_images)
            np.save(args.output_directory + '/part_face_images_' + str(batch) + '.npy', par_images)
            np.save(args.output_directory + '/negative_images_' + str(batch) + '.npy', neg_images)
            pos_images = list()
            par_images = list()
            neg_images = list()

            # Save the bounding box annotations
            par_bboxes = np.asarray(par_bboxes)
            pos_bboxes = np.asarray(pos_bboxes)
            np.save(args.output_directory + '/part_faces_' + str(batch) + '.npy', par_bboxes)
            np.save(args.output_directory + '/positives_' + str(batch) + '.npy', pos_bboxes)
            pos_bboxes = list()
            par_bboxes = list()

            batch = batch + 1

    # Always close the file
    f.close()

    # Save the images
    pos_images = np.asarray(pos_images)
    par_images = np.asarray(par_images)
    neg_images = np.asarray(neg_images)
    np.save(args.output_directory + '/positive_images_' + str(batch) + '.npy', pos_images)
    np.save(args.output_directory + '/part_face_images_' + str(batch) + '.npy', par_images)
    np.save(args.output_directory + '/negative_images_' + str(batch) + '.npy', neg_images)

    # Save the bounding box annotations
    par_bboxes = np.asarray(par_bboxes)
    pos_bboxes = np.asarray(pos_bboxes)
    np.save(args.output_directory + '/part_faces_' + str(batch) + '.npy', par_bboxes)
    np.save(args.output_directory + '/positives_' + str(batch) + '.npy', pos_bboxes)

# def inspect(x):
#     print(type(x))
#     print(x.shape)
#     print(x.dtype)
#     exit()

def show(img, c_bboxes):
    h, w = img.shape[:2]
    print(h, w)
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = w / dpi, h / dpi
    figure, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    for bbox in c_bboxes:
        bb = matplotlib.patches.Rectangle(xy=(bbox[1], bbox[0]), width=bbox[3] - bbox[1], height=bbox[2] - bbox[0], fill=False, edgecolor='r')
        ax.add_patch(bb)
    plt.axis('off')
    plt.show()

def square(bboxes):
    """
    Turn the bounding boxes into squares.
    Args:
    - bboxes: (n, 4) tensor with each row encoded as [y1, x1, y2, x2]
    Return:
    - square_bboxes: (n, 4) tensor of squared bounding boxes
    """
    h = bboxes[:, 2] - bboxes[:, 0]
    w = bboxes[:, 3] - bboxes[:, 1]
    size = tf.math.maximum(h, w)
    x_margin = (size - w) / 2
    y_margin = (size - h) / 2
    y1 = bboxes[:, 0] - y_margin
    x1 = bboxes[:, 1] - x_margin
    y2 = bboxes[:, 2] + y_margin
    x2 = bboxes[:, 3] + x_margin
    square_bboxes = tf.stack([y1, x1, y2, x2], axis=1)
    return square_bboxes

if __name__ == "__main__":
    main()