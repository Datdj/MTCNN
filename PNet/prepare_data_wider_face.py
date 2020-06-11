import sys
sys.path.insert(0, '..')
import argparse
import cv2
import numpy as np
from os import path
import os
from numpy.random import default_rng
from tqdm import tqdm
from utils.misc import create_bbr_annotation, crop_and_resize, find_iou

def main():

    parser = argparse.ArgumentParser(description='Create Positives, Negatives and Part Faces from the WIDER FACE dataset for P-Net.')
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
    args = parser.parse_args()

    # Instantiate a random generator
    rng = default_rng()

    # Prepare for saving
    if not path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # These are for storing the images
    pos_images = list()
    par_images = list()
    neg_images = list()

    # These are for storing the annotations of bounding box regression task
    pos_bboxes = list()
    par_bboxes = list()

    # Start reading the file
    window = np.zeros(3, np.int32)
    f = open(args.annotation_file)
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
        bboxes = np.zeros((num_faces, 4), np.int32)
        invalid = []
        for j in range(num_faces):
            face = f.readline()
            if face[-7] == '1':
                invalid.append(j)
                continue
            bboxes[j] = face[:-14].split(' ')

        # Remove invalid faces
        if len(invalid) > 0:
            bboxes = np.delete(bboxes, invalid, 0)
            if bboxes.shape[0] == 0:
                continue
        num_faces = bboxes.shape[0]

        # Read the image
        img = cv2.imread(args.images_directory + '/' + line[:-1])[:, :, ::-1]

        # Create positives, part faces and negatives
        for bbox in bboxes:
            found_part_face = False
            found_positive = False
            find_negative = False
            for i in range(30):
                if found_positive == True and found_part_face == True:
                    break
                window[2] = max(bbox[2], bbox[3])
                window[2] += ((rng.random() * 0.2 - 0.1) * window[2]).astype(np.int32)
                window[2] = max(window[2], 12)
                max_y, max_x = np.array(img.shape[:-1]) - window[2]
                if max_x < 0 or max_y < 0:
                    continue
                bbox_center = bbox[:2] + bbox[2:] // 2
                if found_part_face == False:
                    window_center = bbox_center + ((rng.random() * 2 - 1) * window[2]).astype(np.int32)
                    window[:2] = np.clip(window_center - window[2] // 2, 0, np.array([max_x, max_y]))
                else:
                    window_center = bbox_center + ((rng.random() * 0.8 - 0.4) * window[2]).astype(np.int32)
                    window[:2] = np.clip(window_center - window[2] // 2, 0, np.array([max_x, max_y]))
                iou = find_iou(window, bbox.reshape(1, 4))[0]

                # If the window is a part face
                if 0.4 <= iou and iou <= 0.65 and found_part_face == False:
                    
                    # Prepare bounding box regression annotation
                    par_bboxes.append(create_bbr_annotation(window, bbox))
                    
                    # Crop out the window and resize if necessary
                    par_images.append(crop_and_resize(img, window, 12))

                    # Move on to the next iteration
                    found_part_face = True
                    continue

                # If the window is a positive
                if iou > 0.65 and found_positive == False:
                    
                    # Prepare bounding box regression annotation
                    pos_bboxes.append(create_bbr_annotation(window, bbox))
                    
                    # Crop out the window and resize if necessary
                    pos_images.append(crop_and_resize(img, window, 12))

                    # Move on to the next iteration
                    found_positive = True
                    continue

                if rng.random() < 0.05 and find_negative == False:
                    find_negative = True
                    if rng.random() < 0.15:
                        ious = find_iou(window, bboxes)
                        if np.max(ious) < 0.3:

                            # Crop out the window and resize if necessary
                            neg_images.append(crop_and_resize(img, window, 12))
            
        # Get 8 negatives out of each image
        num_neg = 0
        for i in range(15):
            window[2] = rng.integers(12, np.min(img.shape[:-1]), endpoint=True)
            max_y, max_x = np.array(img.shape[:-1]) - window[2]
            window[0] = rng.integers(0, max_x + 1)
            window[1] = rng.integers(0, max_y + 1)
            ious = find_iou(window, bboxes)

            if np.max(ious) < 0.3:

                # Crop out the window and resize if necessary
                neg_images.append(crop_and_resize(img, window, 12))

                num_neg += 1

            if num_neg == 8:
                break

    # Always close the file
    f.close()

    # Save the images
    images = np.concatenate((np.asarray(pos_images), np.asarray(par_images), np.asarray(neg_images)), axis=0)
    np.save(args.output_directory + '/pos_par_neg_images.npy', images)

    # Save the bounding box annotations
    np.save(args.output_directory + '/part_faces.npy', np.asarray(par_bboxes))
    np.save(args.output_directory + '/positives.npy', np.asarray(pos_bboxes))

if __name__ == "__main__":
    main()