import os
from os import path
import argparse
import numpy as np
from numpy.random import default_rng
import pandas as pd

def main():
    
    parser = argparse.ArgumentParser(description='Compile data for P-Net training.')
    parser.add_argument(
        '-d',
        '--data-folder',
        type=str,
        help='The path to the folder "data/temp"',
        required=True
    )
    args = parser.parse_args()

    # Prepare saving folders
    if not path.exists(args.data_folder + '/../train'):
        os.mkdir(args.data_folder + '/../train')
    if not path.exists(args.data_folder + '/../validation'):
        os.mkdir(args.data_folder + '/../validation')
    if not path.exists(args.data_folder + '/../test'):
        os.mkdir(args.data_folder + '/../test')

    # Load all the files
    pos_par_neg_images = np.load(args.data_folder + '/pos_par_neg_images.npy')
    lm_images = np.load(args.data_folder + '/lm_images.npy')
    pos_annotation = np.load(args.data_folder + '/positives.npy')
    par_annotation = np.load(args.data_folder + '/part_faces.npy')
    lm_annotation = np.load(args.data_folder + '/landmarks.npy')
    pos_origin = pd.read_excel(args.data_folder + '/pos_origin.xlsx')
    par_origin = pd.read_excel(args.data_folder + '/par_origin.xlsx')
    neg_origin = pd.read_excel(args.data_folder + '/neg_origin.xlsx')
    lm_origin = pd.read_excel(args.data_folder + '/lm_origin.xlsx')   
    num_pos = pos_annotation.shape[0]
    num_par = par_annotation.shape[0]
    num_neg = pos_par_neg_images.shape[0] - num_pos - num_par
    num_lm = lm_annotation.shape[0]

    # Merge all images into one big array
    images = np.concatenate((pos_par_neg_images, lm_images), 0)
    num_images = images.shape[0]

    # Merge all original windows into one big file
    origins = pd.concat([pos_origin, par_origin, neg_origin, lm_origin], ignore_index=True).drop(columns=['Unnamed: 0']).astype(dtype={'file_name': str, 'x': np.uint16, 'y': np.uint16, 'window_size': np.uint16})

    # Create classification labels
    pos_class = np.concatenate((np.ones((num_pos, 2), dtype=np.int8), np.zeros((num_pos, 1), dtype=np.int8)), axis=1)
    par_class = np.zeros((num_par, 3), dtype=np.int8)
    neg_class = np.concatenate((np.ones((num_neg, 1), dtype=np.int8), np.zeros((num_neg, 1), dtype=np.int8), np.ones((num_neg, 1), dtype=np.int8)), axis=1)
    lm_class = np.zeros((num_lm, 3), dtype=np.int8)
    class_labels = np.concatenate((pos_class, par_class, neg_class, lm_class), axis=0)

    # Create bounding box regression labels
    pos_bb = np.concatenate((np.ones((num_pos, 1)), pos_annotation), axis=1)
    par_bb = np.concatenate((np.ones((num_par, 1)), par_annotation), axis=1)
    neg_bb = np.zeros((num_neg, 5))
    lm_bb = np.zeros((num_lm, 5))
    bb_labels = np.concatenate((pos_bb, par_bb, neg_bb, lm_bb), axis=0)

    # Create landmark regression labels
    pos_lm = np.zeros((num_pos, 11))
    par_lm = np.zeros((num_par, 11))
    neg_lm = np.zeros((num_neg, 11))
    lm_lm = np.concatenate((np.ones((num_lm, 1)), lm_annotation), axis=1)
    lm_labels = np.concatenate((pos_lm, par_lm, neg_lm, lm_lm), axis=0)

    # Shuffle everything
    rng = default_rng()
    random_indices = rng.permutation(num_images)
    images = images[random_indices]
    class_labels = class_labels[random_indices]
    bb_labels = bb_labels[random_indices]
    lm_labels = lm_labels[random_indices]
    origins = origins.iloc[random_indices].reset_index(drop=True)

    # Split data into train, validation and test sets
    split1 = round(0.9 * num_images)
    split2 = round(0.95 * num_images)
    train_images = images[:split1]
    train_class_labels = class_labels[:split1]
    train_bb_labels = bb_labels[:split1]
    train_lm_labels = lm_labels[:split1]
    train_origins = origins[:split1]
    val_images = images[split1:split2]
    val_class_labels = class_labels[split1:split2]
    val_bb_labels = bb_labels[split1:split2]
    val_lm_labels = lm_labels[split1:split2]
    val_origins = origins[split1:split2].reset_index(drop=True)
    test_images = images[split2:]
    test_class_labels = class_labels[split2:]
    test_bb_labels = bb_labels[split2:]
    test_lm_labels = lm_labels[split2:]
    test_origins = origins[split2:].reset_index(drop=True)

    # Save everything
    np.save(args.data_folder + '/../train/images.npy', train_images)
    np.save(args.data_folder + '/../train/class_labels.npy', train_class_labels)
    np.save(args.data_folder + '/../train/bounding_box_labels.npy', train_bb_labels)
    np.save(args.data_folder + '/../train/landmark_labels.npy', train_lm_labels)
    train_origins.to_excel(args.data_folder + '/../train/origins.xlsx')
    np.save(args.data_folder + '/../validation/images.npy', val_images)
    np.save(args.data_folder + '/../validation/class_labels.npy', val_class_labels)
    np.save(args.data_folder + '/../validation/bounding_box_labels.npy', val_bb_labels)
    np.save(args.data_folder + '/../validation/landmark_labels.npy', val_lm_labels)
    val_origins.to_excel(args.data_folder + '/../validation/origins.xlsx')
    np.save(args.data_folder + '/../test/images.npy', test_images)
    np.save(args.data_folder + '/../test/class_labels.npy', test_class_labels)
    np.save(args.data_folder + '/../test/bounding_box_labels.npy', test_bb_labels)
    np.save(args.data_folder + '/../test/landmark_labels.npy', test_lm_labels)
    test_origins.to_excel(args.data_folder + '/../test/origins.xlsx')

if __name__ == "__main__":
    main()