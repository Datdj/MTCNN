import argparse
import cv2
import numpy as np
import pandas as pd
from os import path
import os
from numpy.random import default_rng
from tqdm import tqdm

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

    rng = default_rng()

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
    #         if rng.random() < 0.1:
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
        os.makedirs(args.output_directory)
    if not path.exists(args.output_directory + '/positives'):
        os.mkdir(args.output_directory + '/positives')
    if not path.exists(args.output_directory + '/negatives'):
        os.mkdir(args.output_directory + '/negatives')
    if not path.exists(args.output_directory + '/part faces'):
        os.mkdir(args.output_directory + '/part faces')

    # These are for storing the images
    pos_images = np.zeros((0, 12, 12, 3), np.uint8)
    par_images = np.zeros((0, 12, 12, 3), np.uint8)
    neg_images = np.zeros((0, 12, 12, 3), np.uint8)

    # These are for storing the annotations of bounding box regression task
    pos_bboxes = np.zeros((0, 4), np.float)
    par_bboxes = np.zeros((0, 4), np.float)

    # These are for storing the original windows
    pos_origin_windows = np.zeros((0, 3), np.int32)
    pos_origin_names = []
    par_origin_windows = np.zeros((0, 3), np.int32)
    par_origin_names = []
    neg_origin_windows = np.zeros((0, 3), np.int32)
    neg_origin_names = []

    # Name the file
    pos_name = 0
    neg_name = 0
    par_name = 0

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
        img = cv2.imread(args.images_directory + '/' + line[:-1])

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
                if 0.4 <= iou and iou <= 0.65 and found_part_face == False:
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
                    par_origin_windows = np.concatenate((par_origin_windows, window.reshape((1, 3))), axis=0)
                    par_origin_names.append(line[:-1])
                    par_images = np.concatenate((par_images, result.reshape((1, 12, 12, 3))), axis=0)
                    cv2.imwrite(
                        args.output_directory + '/part faces/par_' + str(par_name).zfill(6) + '.jpg',
                        result
                    )
                    par_name += 1
                    found_part_face = True
                    continue
                if iou > 0.65 and found_positive == False:
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
                    pos_origin_windows = np.concatenate((pos_origin_windows, window.reshape((1, 3))), axis=0)
                    pos_origin_names.append(line[:-1])
                    pos_images = np.concatenate((pos_images, result.reshape((1, 12, 12, 3))), axis=0)
                    cv2.imwrite(
                        args.output_directory + '/positives/pos_' + str(pos_name).zfill(6) + '.jpg',
                        result
                    )
                    pos_name += 1
                    found_positive = True
                    continue
                if rng.random() < 0.05 and find_negative == False:
                    find_negative = True
                    if rng.random() < 0.15:
                        ious = find_iou(window, bboxes)
                        if np.max(ious) < 0.15:
                            # Resize the cropped image to 12 x 12 if necessary
                            result = img[window[1]:window[1] + window[2], window[0]:window[0] + window[2], :]
                            if window[2] > 12:
                                result = cv2.resize(result, (12, 12), interpolation=cv2.INTER_AREA)
                            # Save the negative
                            neg_origin_windows = np.concatenate((neg_origin_windows, window.reshape((1, 3))), axis=0)
                            neg_origin_names.append(line[:-1])
                            neg_images = np.concatenate((neg_images, result.reshape((1, 12, 12, 3))), axis=0)
                            cv2.imwrite(
                                args.output_directory + '/negatives/neg_' + str(neg_name).zfill(6) + '.jpg',
                                result
                            )
                            neg_name += 1
            
        # Get 8 negatives out of each image
        num_neg = 0
        for i in range(15):
            window[2] = rng.integers(12, np.min(img.shape[:-1]), endpoint=True)
            max_y, max_x = np.array(img.shape[:-1]) - window[2]
            window[0] = rng.integers(0, max_x + 1)
            window[1] = rng.integers(0, max_y + 1)
            ious = find_iou(window, bboxes)
            if np.max(ious) < 0.15:
                # Resize the cropped image to 12 x 12 if necessary
                result = img[window[1]:window[1] + window[2], window[0]:window[0] + window[2], :]
                if window[2] > 12:
                    result = cv2.resize(result, (12, 12), interpolation=cv2.INTER_AREA)
                # Save the negative
                neg_origin_windows = np.concatenate((neg_origin_windows, window.reshape((1, 3))), axis=0)
                neg_origin_names.append(line[:-1])
                neg_images = np.concatenate((neg_images, result.reshape((1, 12, 12, 3))), axis=0)
                cv2.imwrite(
                    args.output_directory + '/negatives/neg_' + str(neg_name).zfill(6) + '.jpg',
                    result
                )
                neg_name += 1
                num_neg += 1
            if num_neg == 8:
                break

    # Always close the file
    f.close()

    # Save the images
    images = np.concatenate((pos_images, par_images, neg_images), axis=0)
    np.save(args.output_directory + '/pos_par_neg_images.npy', images)

    # Save the bounding box annotations
    np.save(args.output_directory + '/part_faces.npy', par_bboxes)
    np.save(args.output_directory + '/positives.npy', pos_bboxes)
    pd.DataFrame(par_bboxes).to_excel(args.output_directory + '/part_faces.xlsx')
    pd.DataFrame(pos_bboxes).to_excel(args.output_directory + '/positives.xlsx')

    # Save the original windows and file names
    pd.DataFrame(
        data={
            'file_name': pos_origin_names,
            'x': pos_origin_windows[:, 0],
            'y': pos_origin_windows[:, 1],
            'window_size': pos_origin_windows[:, 2]
        }
    ).to_excel(args.output_directory + '/pos_origin.xlsx')
    pd.DataFrame(
        data={
            'file_name': par_origin_names,
            'x': par_origin_windows[:, 0],
            'y': par_origin_windows[:, 1],
            'window_size': par_origin_windows[:, 2]
        }
    ).to_excel(args.output_directory + '/par_origin.xlsx')
    pd.DataFrame(
        data={
            'file_name': neg_origin_names,
            'x': neg_origin_windows[:, 0],
            'y': neg_origin_windows[:, 1],
            'window_size': neg_origin_windows[:, 2]
        }
    ).to_excel(args.output_directory + '/neg_origin.xlsx')

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