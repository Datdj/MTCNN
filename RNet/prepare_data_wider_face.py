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
from PNet.model import pnet_predict
import tensorflow as tf
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
        '--model-path',
        type=str,
        help='The path to the .hdf5 file containing the weights of P-Net',
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
    pnet = pnet_predict()
    pnet.load_weights(filepath=args.model_path, by_name=True)

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

        # Remove invalid faces
        if len(invalid) > 0:
            ground_truth = np.delete(ground_truth, invalid, 0)
            if ground_truth.shape[0] == 0:
                continue
        num_faces = ground_truth.shape[0]

        # Read the image
        img = cv2.imread(args.images_directory + '/' + line[:-1])[:, :, ::-1]
        # print(img.shape)

        # Get the original image size
        height, width = img.shape[:2]

        # Initialize the scale
        scale = 12 / 20
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
        c_bboxes = tf.concat(c_bboxes, axis=0)
        confidence = tf.concat(confidence, axis=0)
        final_indices = tf.image.non_max_suppression(c_bboxes, confidence, confidence.shape[0], 0.7)
        c_bboxes = tf.gather(c_bboxes, final_indices)
        # print(c_bboxes.shape)

        # Round out the values
        c_bboxes = tf.cast(tf.round(c_bboxes), tf.int32)

        c_bboxes = c_bboxes.numpy()[:, :3]
        c_bboxes[:, 2] = c_bboxes[:, 2] - c_bboxes[:, 0]

        temp = ground_truth[:, 0].copy()
        ground_truth[:, 0] = ground_truth[:, 1]
        ground_truth[:, 1] = temp
        temp = ground_truth[:, 2].copy()
        ground_truth[:, 2] = ground_truth[:, 3]
        ground_truth[:, 3] = temp

        # print(ground_truth.shape)
        # print(c_bboxes.shape)

        # Calculate iou
        ious = find_iou_bulk(c_bboxes, ground_truth)

        # print(c_bboxes.shape[0])
        # print(np.count_nonzero(ious > 0.65))
        # exit()

        

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
                pos_images.append(crop_and_resize_v2(img, bbox, 24))
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
            limit = 50
            size = []
            indices = rng.permutation(np.arange(num_windows)[neg_mask])
            # print(indices)
            for j in indices:
                if c_bboxes[j, 2] not in size:
                    size.append(c_bboxes[j, 2])
                    # print(c_bboxes[j])
                    count = count + 1
                    neg_images.append(crop_and_resize_v2(img, c_bboxes[j], 24))
                if count == limit:
                    break

        # # Loop through all windows
        # for i in range(num_windows):
            
        #     # If this window is a positive
        #     if max_ious[i] > 0.65:

        #         # Create bbr annotation
        #         pos_bboxes.append(create_bbr_annotation_v2(c_bboxes[i], ground_truth[max_iou_mask[i]]))
        #         print(pos_bboxes)
        #         exit()
                
        #         # Crop out the window and resize if necessary
        #         pos_images = np.concatenate((pos_images, crop_and_resize(img, c_bboxes[i], 24)), axis=0)

            # # If this window is a part face
            # elif 0.45 <= max_ious[i] and max_ious[i] <= 0.65:

            #     # Create bbr annotation
            #     par_bboxes = np.concatenate((par_bboxes, create_bbr_annotation(c_bboxes[i], ground_truth[max_iou_mask[i]])), axis=0)
                
            #     # Crop out the window and resize if necessary
            #     par_images = np.concatenate((par_images, crop_and_resize(img, c_bboxes[i], 24)), axis=0)

            # # If this window is a negative
            # elif max_ious[i] < 0.3:

            #     # Crop out the window and resize if necessary
            #     neg_images = np.concatenate((neg_images, crop_and_resize(img, c_bboxes[i], 24)), axis=0)
        
        if (i + 1) % 1000 == 0:

            # Save the images
            pos_images = np.asarray(pos_images)
            par_images = np.asarray(par_images)
            neg_images = np.asarray(neg_images)
            np.save(args.output_directory + '/positive_images_' + str(batch) + '.npy', pos_images)
            np.save(args.output_directory + '/part_face_images' + str(batch) + '.npy', par_images)
            np.save(args.output_directory + '/negative_images' + str(batch) + '.npy', neg_images)
            pos_images = list()
            par_images = list()
            neg_images = list()

            # Save the bounding box annotations
            par_bboxes = np.asarray(par_bboxes)
            pos_bboxes = np.asarray(pos_bboxes)
            np.save(args.output_directory + '/part_faces' + str(batch) + '.npy', par_bboxes)
            np.save(args.output_directory + '/positives' + str(batch) + '.npy', pos_bboxes)
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
    np.save(args.output_directory + '/part_face_images' + str(batch) + '.npy', par_images)
    np.save(args.output_directory + '/negative_images' + str(batch) + '.npy', neg_images)

    # Save the bounding box annotations
    par_bboxes = np.asarray(par_bboxes)
    pos_bboxes = np.asarray(pos_bboxes)
    np.save(args.output_directory + '/part_faces' + str(batch) + '.npy', par_bboxes)
    np.save(args.output_directory + '/positives' + str(batch) + '.npy', pos_bboxes)

if __name__ == "__main__":
    main()