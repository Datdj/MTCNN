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

    # Prepare for saving
    if not path.exists(args.output_directory):
        os.mkdir(args.output_directory)
    if not path.exists(args.output_directory + '/positives'):
        os.mkdir(args.output_directory + '/positives')
    if not path.exists(args.output_directory + '/negatives'):
        os.mkdir(args.output_directory + '/negatives')
    if not path.exists(args.output_directory + '/part faces'):
        os.mkdir(args.output_directory + '/part faces')

    # Open the annotation file
    f = open(args.annotation_file)

    # These are for storing the annotations of bounding box regression task
    pos_bboxes = np.zeros((0, 4), np.float)
    par_bboxes = np.zeros((0, 4), np.float)

    # Name the file
    pos_name = 0
    neg_name = 0
    par_name = 0

    # Start reading the file
    count = 0
    window = np.zeros(3, np.int32)
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
            if bboxes.shape[0] == 0:
                continue

        # Read the image
        img = cv2.imread(args.images_directory + '/' + line[:-1])

        # For every ground truth face, create a positive, a part face and a negative
        for bbox in bboxes:
            window[2] = max(bbox[2], bbox[3], 12)
            max_y, max_x = np.array(img.shape[:-1]) - window[2]

            # See 0--Parade/0_Parade_Parade_0_939.jpg
            if max_x < 0 or max_y < 0:
                continue

            # Part Face
            if bbox[2] * bbox[3] >= window[2] ** 2 * 0.4:
                for i in range(10):
                    window[0] = max(0, min(bbox[0] + ((np.random.random() * 2 - 1) * window[2]).astype(np.int32), max_x))
                    window[1] = max(0, min(bbox[1] + ((np.random.random() * 2 - 1) * window[2]).astype(np.int32), max_y))
                    iou = find_iou(window, bbox.reshape(1, 4))[0]
                    if 0.4 <= iou and iou <= 0.65:

                        # Prepare bounding box regression annotation
                        top_left = np.maximum(window[:2], bbox[:2]) - window[:2]
                        bottom_right = np.minimum(window[:2] + window[2] - 1, bbox[:2] + bbox[2:] - 1) - window[:2]
                        par_bbox = (np.append(top_left, bottom_right - top_left) / (window[2] - 1)).reshape((1,4))
                        par_bboxes = np.append(par_bboxes, par_bbox, 0)
                        
                        # Resize the cropped image to 12 x 12 if necessary
                        result = img[window[1]:window[1] + window[2], window[0]:window[0] + window[2], :]
                        if window[2] > 12:
                            result = cv2.resize(result, (12, 12), interpolation=cv2.INTER_AREA)

                        # Save the part face
                        cv2.imwrite(
                            args.output_directory + '/part faces/par_' + str(par_name).zfill(6) + '.jpg',
                            result
                        )
                        par_name += 1
                        break

                # Positive
                if bbox[2] * bbox[3] > window[2] ** 2 * 0.65:
                    for i in range(10):
                        window[0] = max(0, min(bbox[0] + ((np.random.random() - 0.5) * window[2]).astype(np.int32), max_x))
                        window[1] = max(0, min(bbox[1] + ((np.random.random() - 0.5) * window[2]).astype(np.int32), max_y))
                        iou = find_iou(window, bbox.reshape(1, 4))[0]
                        if iou > 0.65:

                            # Prepare bounding box regression annotation
                            top_left = np.maximum(window[:2], bbox[:2]) - window[:2]
                            bottom_right = np.minimum(window[:2] + window[2] - 1, bbox[:2] + bbox[2:] - 1) - window[:2]
                            pos_bbox = (np.append(top_left, bottom_right - top_left) / (window[2] - 1)).reshape((1,4))
                            pos_bboxes = np.append(pos_bboxes, pos_bbox, 0)

                            # Resize the cropped image to 12 x 12 if necessary
                            result = img[window[1]:window[1] + window[2], window[0]:window[0] + window[2], :]
                            if window[2] > 12:
                                result = cv2.resize(result, (12, 12), interpolation=cv2.INTER_AREA)

                            # Save the positive
                            cv2.imwrite(
                                args.output_directory + '/positives/pos_' + str(pos_name).zfill(6) + '.jpg',
                                result
                            )
                            pos_name += 1
                            break
            
            # Negative
            for i in range(10):
                window[0] = np.random.randint(0, max_x + 1)
                window[1] = np.random.randint(0, max_y + 1)
                ious = find_iou(window, bboxes)
                if np.max(ious) < 0.3:

                    # Resize the cropped image to 12 x 12 if necessary
                    result = img[window[1]:window[1] + window[2], window[0]:window[0] + window[2], :]
                    if window[2] > 12:
                        result = cv2.resize(result, (12, 12), interpolation=cv2.INTER_AREA)

                    # Save the negative
                    cv2.imwrite(
                        args.output_directory + '/negatives/neg_' + str(neg_name).zfill(6) + '.jpg',
                        result
                    )
                    neg_name += 1
                    break

        # Print out the progress
        count += 1
        print(str(count) + '/12880 - ' + str(round(count / 12880 * 100)) + '%', end='\r')

    # Always close the file
    f.close()

    # Save the bounding box annotations
    pd.DataFrame(par_bboxes).to_excel(args.output_directory + '/part_faces.xlsx')
    pd.DataFrame(pos_bboxes).to_excel(args.output_directory + '/positives.xlsx')

# Calculate the Intersection over Union
def find_iou(window, bboxes):
    lowest = np.minimum(bboxes[:, :2], window[:2])
    highest = np.maximum(bboxes[:, :2] + bboxes[:, 2:], window[:2] + window[2])
    intersection = highest - lowest - bboxes[:, 2:] - window[2]
    is_not_overlapped = intersection >= 0
    intersection[is_not_overlapped] = 0
    intersection *= -1
    intersection = intersection[:, 0] * intersection[:, 1]
    union = bboxes[:, 2] * bboxes[:, 3] + window[2] ** 2 - intersection
    return intersection / union

if __name__ == "__main__":
    main()