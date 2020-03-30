import argparse
import cv2
import numpy as np
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description='Create Positives, Negatives and Part Faces from the WIDER FACE dataset for P-Net.')
    parser.add_argument(
        '-a',
        '--annotation-file',
        type=str,
        help='The path to the annotation file of the WIDER FACE dataset'
    )
    parser.add_argument(
        '-i',
        '--images-directory',
        type=str,
        help='The path to the folder that contains WIDER FACE dataset'
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
    #     # img = cv2.imread(args.images_directory + '/' + line[:-1])
    #     num_faces = int(f.readline())
    #     if num_faces == 0:
    #         f.readline()
    #         continue
    #     bboxes = np.zeros((num_faces, 10), np.uint16)
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
    # pd.DataFrame(stats).to_excel('../../Raw data/WIDER_train/stats.xlsx')

if __name__ == "__main__":
    main()