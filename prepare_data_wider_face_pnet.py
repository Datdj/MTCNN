import argparse
import cv2
import numpy as np
import pandas as pd
from os import path
import os

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
        required=True
    )
    args = parser.parse_args()

    # # Get some statistics
    # stats = {
    #     'clear': 0,
    #     'normal blur': 0,
    #     'heavy blur': 0,
    #     'typical expression': 0,
    #     'exaggerate expression': 0,
    #     'normal illumination': 0,
    #     'extreme illumination': 0,
    #     'no occlusion': 0,
    #     'partial occlusion': 0,
    #     'heavy occlusion': 0,
    #     'typical pose': 0,
    #     'atypical pose': 0,
    #     'valid': 0,
    #     'invalid': 0,
    #     'perfect': [0]
    # }
    # f = open(args.annotation_file)
    # while True:
    #     line = f.readline()
    #     if line == '':
    #         break
    #     num_faces = int(f.readline())
    #     if num_faces == 0:
    #         f.readline()
    #         continue
    #     bboxes = np.zeros((num_faces, 10), np.int32)
    #     for i in range(num_faces):
    #         bboxes[i] = f.readline()[:-2].split(' ')
    #         # Check whether the face is perfect or not and count
    #         if np.count_nonzero(bboxes[i, 4:]) == 0:
    #             stats['perfect'][0] += 1
    #             stats['clear'] += 1
    #             stats['no occlusion'] += 1
    #             continue
    #         # Count the blur faces
    #         if bboxes[i, 4] == 0:
    #             stats['clear'] += 1
    #         elif bboxes[i, 4] == 1:
    #             stats['normal blur'] += 1
    #         else:
    #             stats['heavy blur'] += 1
    #         # Count the faces with occlusion
    #         if bboxes[i, 8] == 0:
    #             stats['no occlusion'] += 1
    #         elif bboxes[i, 8] == 1:
    #             stats['partial occlusion'] += 1
    #         else:
    #             stats['heavy occlusion'] += 1
    #     # Count the expressions
    #     num_ex_expressions = np.count_nonzero(bboxes[:, 5])
    #     stats['exaggerate expression'] += num_ex_expressions
    #     stats['typical expression'] += num_faces - num_ex_expressions
    #     # Count the faces with illumination
    #     num_ex_illumination = np.count_nonzero(bboxes[:, 6])
    #     stats['extreme illumination'] += num_ex_illumination
    #     stats['normal illumination'] += num_faces - num_ex_illumination
    #     # Pose
    #     num_atyp_pose = np.count_nonzero(bboxes[:, 9])
    #     stats['atypical pose'] += num_atyp_pose
    #     stats['typical pose'] += num_faces - num_atyp_pose
    #     # Invalid
    #     num_invalid = np.count_nonzero(bboxes[:, 7])
    #     stats['invalid'] += num_invalid
    #     stats['valid'] += num_faces - num_invalid
    # f.close()
    # pd.DataFrame(stats).to_excel('stats.xlsx')

    # # Print out some images with invalid faces for inspection
    # count = 0
    # f = open(args.annotation_file)
    # while True:
    #     invalid = []
    #     line = f.readline()
    #     if line == '':
    #         break
    #     num_faces = int(f.readline())
    #     if num_faces == 0:
    #         f.readline()
    #         continue
    #     bboxes = np.zeros((num_faces, 10), np.int32)
    #     for i in range(num_faces):
    #         bboxes[i] = f.readline()[:-2].split(' ')
    #         if bboxes[i, 7] == 1:
    #             invalid.append(i)
    #     if len(invalid) > 0:
    #         if np.random.random() < 0.1:
    #             count += 1
    #             img = cv2.imread(args.images_directory + '/' + line[:-1])
    #             for i in range(num_faces):
    #                 top_left = (bboxes[i, 0], bboxes[i, 1])
    #                 bottom_right = (bboxes[i, 0] + bboxes[i, 2], bboxes[i, 1] + bboxes[i, 3])
    #                 if i in invalid:
    #                     cv2.rectangle(img, top_left, bottom_right, (0, 0, 255))
    #                 else:
    #                     cv2.rectangle(img, top_left, bottom_right, (0, 255, 0))
    #             cv2.imshow('image', img)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #     if count == 10:
    #         break
    # f.close()

    # Open the annotation file
    f = open(args.annotation_file)
    
    # Use this to name the output images
    count = 0

    # These are for storing the annotations of bounding box regression task
    pos_bboxes = np.zeros((0, 4), np.uint8)
    par_bboxes = np.zeros((0, 4), np.uint8)

    # Start reading the file
    while True:
        
        # Read the file line by line
        line = f.readline()
        
        # Stop when reaching end of file
        if line == '':
            break

        # Obtain the number of faces in this image
        num_faces = int(f.readline())

        # Skip the image if it does not contain any faces
        if num_faces == 0:
            f.readline()
            continue
        
        # Obtain bounding boxes while taking into account invalid faces
        bboxes = np.zeros((num_faces, 4), np.int32)
        invalid = []
        for i in range(num_faces):
            face = f.readline()
            if face[-7] == '1':
                invalid.append(i)
                continue
            bboxes[i] = face[:-14].split(' ')

        # Remove invalid faces
        if len(invalid) > 0:
            bboxes = np.delete(bboxes, invalid, 0)

        # Calculate the window size for cropping
        crop_size = max(np.max(bboxes[:, 2:4]), 12)

        # Skip if not a single face can cover more than 65% of the window (meaning no positive for this one)
        if np.max(bboxes[:, 2] * bboxes[:, 3]) <= crop_size * crop_size * 0.65:
            count += 1
            continue

        # Read the image
        img = cv2.imread(args.images_directory + '/' + line[:-1])

        # Prepare for saving
        if not path.exists(args.output_directory):
            os.mkdir(args.output_directory)
        if not path.exists(args.output_directory + '/positives'):
            os.mkdir(args.output_directory + '/positives')
        if not path.exists(args.output_directory + '/negatives'):
            os.mkdir(args.output_directory + '/negatives')
        if not path.exists(args.output_directory + '/part faces'):
            os.mkdir(args.output_directory + '/part faces')

        # These are the upper limits of the top left point of the window
        max_y, max_x = np.array(img.shape[:-1]) - crop_size + 1

        # Skip this image if max_x or max_y <= 0 (See 0--Parade/0_Parade_Parade_0_939.jpg)
        if max_x <= 0 or max_y <= 0:
            count += 1
            continue

        # Use this to make sure we have 1 of each kind of annotation from this image
        okay = np.zeros(3, bool)

        # Skip if not a single face can cover more than 65% of the window (meaning no positive for this one)
        if np.max(bboxes[:, 2] * bboxes[:, 3]) <= crop_size * crop_size * 0.65:
            okay[2] = True

        # Start cropping
        num_tries = 0
        while True:

            # Create a random window
            window = np.zeros(4, np.int32)
            window[0] = np.random.randint(0, max_x)
            window[1] = np.random.randint(0, max_y)
            window[2:4] = crop_size

            # Calculate the Intersection over Union (IoU) of the window with all the faces in the image
            lowest = np.minimum(bboxes[:, :2], window[:2])
            highest = np.maximum(bboxes[:, :2] + bboxes[:, 2:], window[:2] + window[2:])
            intersection = highest - lowest - bboxes[:, 2:] - window[2:]
            is_not_overlapped = intersection >= 0
            intersection[is_not_overlapped] = 0
            intersection *= -1
            intersection = intersection[:, 0] * intersection[:, 1]
            union = bboxes[:, 2] * bboxes[:, 3] + window[2] * window[3] - intersection
            iou = intersection / union

            # Check if the window is a negative
            if np.max(iou) < 0.3 and okay[0] == False:
                okay[0] = True

                # Resize the cropped image to 12 x 12
                if crop_size > 12:
                    resized = cv2.resize(img[window[1]:window[1] + window[3], window[0]:window[0] + window[2], :],
                    (12, 12),
                    interpolation=cv2.INTER_AREA
                )

                # Save the negative
                cv2.imwrite(
                    args.output_directory + '/negatives/neg_' + str(count) + '.jpg',
                    resized
                )
            
            # or if the window is a part face
            elif 0.4 <= np.max(iou) and np.max(iou) <= 0.65 and okay[1] == False:
                okay[1] = True

                # Prepare bounding box regression annotation
                top_left = np.maximum(window[:2], bboxes[np.argmax(iou), :2]) - window[:2]
                bottom_right = np.minimum(window[:2] + crop_size, bboxes[np.argmax(iou), :2] + bboxes[np.argmax(iou), 2:]) - window[:2]
                par_bbox = (np.append(top_left, bottom_right - top_left) / crop_size * 12).astype(np.uint8).reshape((1,4))
                par_bboxes = np.append(par_bboxes, par_bbox, 0)
                
                # Resize the cropped image to 12 x 12
                if crop_size > 12:
                    resized = cv2.resize(img[window[1]:window[1] + window[3], window[0]:window[0] + window[2], :],
                    (12, 12),
                    interpolation=cv2.INTER_AREA
                )

                # Save the part face
                cv2.imwrite(
                    args.output_directory + '/part faces/par_' + str(count) + '.jpg',
                    resized
                )
                
            # or if the window is a positive
            elif np.max(iou) > 0.65 and okay[2] == False:
                okay[2] = True

                # Prepare bounding box regression annotation
                top_left = np.maximum(window[:2], bboxes[np.argmax(iou), :2]) - window[:2]
                bottom_right = np.minimum(window[:2] + crop_size, bboxes[np.argmax(iou), :2] + bboxes[np.argmax(iou), 2:]) - window[:2]
                pos_bbox = (np.append(top_left, bottom_right - top_left) / crop_size * 12).astype(np.uint8).reshape((1,4))
                pos_bboxes = np.append(pos_bboxes, pos_bbox, 0)

                # Resize the cropped image to 12 x 12
                if crop_size > 12:
                    resized = cv2.resize(img[window[1]:window[1] + window[3], window[0]:window[0] + window[2], :],
                    (12, 12),
                    interpolation=cv2.INTER_AREA
                )

                # Save the positive
                cv2.imwrite(
                    args.output_directory + '/positives/pos_' + str(count) + '.jpg',
                    resized
                )
            
            # Stop the loop if we have enough annotations from this image
            if np.count_nonzero(okay) == 3 or num_tries == 500000:
                break
            
            # Next try
            num_tries += 1
        
        # Let's move on to the next image
        count += 1

    # Always close the file
    f.close()

    # Save the bounding box annotations
    pd.DataFrame(par_bboxes).to_excel(args.output_directory + '/part faces/part_faces.xlsx')
    pd.DataFrame(pos_bboxes).to_excel(args.output_directory + '/positives/positives.xlsx')

if __name__ == "__main__":
    main()