import numpy as np
import argparse
from numpy.random import default_rng
from os import path
import os

def get_arguments():
    """
    Get arguments from the command line.
    """
    parser = argparse.ArgumentParser(description='Compile data for R-Net training.')
    parser.add_argument(
        '-d',
        '--data-folder',
        type=str,
        help='The path to the folder "data/temp"',
        required=True
    )
    return parser.parse_args()

def main():
    rng = default_rng()
    args = get_arguments()
    in_dir = args.data_folder

    # Prepare saving folders
    if not path.exists(in_dir + '/../train'):
        os.mkdir(in_dir + '/../train')
    if not path.exists(in_dir + '/../validation'):
        os.mkdir(in_dir + '/../validation')
    if not path.exists(in_dir + '/../test'):
        os.mkdir(in_dir + '/../test')

    # Load positives
    images = list()
    bb = list()
    images.append(np.load(in_dir + '/pos_images_extra.npy'))
    bb.append(np.load(in_dir + '/positives_extra.npy'))
    for i in range(8):
        images.append(np.load(in_dir + '/positive_images_' + str(i) + '.npy'))
        bb.append(np.load(in_dir + '/positives_' + str(i) + '.npy'))
    images = np.concatenate(images)
    bb = np.concatenate(bb)
    num_img = images.shape[0]

    # Shuffle positives
    random_indices = rng.permutation(num_img)
    images = images[random_indices]
    bb = bb[random_indices]

    # Pad 0 to positives bounding box labels
    bb = np.concatenate((np.ones((num_img, 1), dtype=np.float32), bb), axis=1)

    # Split positives into train, validation and test
    val_pos_img = images[:9500]
    val_pos_bb = bb[:9500]
    test_pos_img = images[9500:19000]
    test_pos_bb = bb[9500:19000]

    # Save positives into files
    temp = (num_img - 19000) // 6
    for i in range(5):
        np.save(in_dir + '/../train/pos_img_' + str(i) + '.npy', images[19000 + i * temp:19000 + (i + 1) * temp])
        np.save(in_dir + '/../train/pos_bb_' + str(i) + '.npy', bb[19000 + i * temp:19000 + (i + 1) * temp])
    np.save(in_dir + '/../train/pos_img_5.npy', images[19000 + 5 * temp:])
    np.save(in_dir + '/../train/pos_bb_5.npy', bb[19000 + 5 * temp:])

    # Load part faces
    images = list()
    bb = list()
    for i in range(8):
        images.append(np.load(in_dir + '/part_face_images_' + str(i) + '.npy'))
        bb.append(np.load(in_dir + '/part_faces_' + str(i) + '.npy'))
    images = np.concatenate(images)
    bb = np.concatenate(bb)
    num_img = images.shape[0]

    # Shuffle part faces
    random_indices = rng.permutation(num_img)
    images = images[random_indices]
    bb = bb[random_indices]

    # Pad 0 to part faces bounding box labels
    bb = np.concatenate((np.ones((num_img, 1), dtype=np.float32), bb), axis=1)

    # Split part faces into train, validation and test
    val_par_img = images[:9500]
    val_par_bb = bb[:9500]
    test_par_img = images[9500:19000]
    test_par_bb = bb[9500:19000]

    # Save part faces into files
    temp = (num_img - 19000) // 3
    for i in range(2):
        np.save(in_dir + '/../train/par_img_' + str(i) + '.npy', images[19000 + i * temp:19000 + (i + 1) * temp])
        np.save(in_dir + '/../train/par_bb_' + str(i) + '.npy', bb[19000 + i * temp:19000 + (i + 1) * temp])
    np.save(in_dir + '/../train/par_img_2.npy', images[19000 + 2 * temp:])
    np.save(in_dir + '/../train/par_bb_2.npy', bb[19000 + 2 * temp:])
    del bb

    # Load negatives
    images = list()
    for i in range(8):
        images.append(np.load(in_dir + '/negative_images_' + str(i) + '.npy'))
    images = np.concatenate(images)
    num_img = images.shape[0]

    # Shuffle negatives
    random_indices = rng.permutation(num_img)
    images = images[random_indices]

    # Split negatives into train, validation and test
    val_neg_img = images[:9500]
    test_neg_img = images[9500:19000]

    # Save negatives into files
    temp = (num_img - 19000) // 3
    for i in range(2):
        np.save(in_dir + '/../train/neg_img_' + str(i) + '.npy', images[19000 + i * temp:19000 + (i + 1) * temp])
    np.save(in_dir + '/../train/neg_img_2.npy', images[19000 + 2 * temp:])

    # Load landmarks
    images = list()
    lm = list()
    for i in range(9):
        images.append(np.load(in_dir + '/lm_images_' + str(i) + '.npy'))
        lm.append(np.load(in_dir + '/landmarks_' + str(i) + '.npy'))
    images = np.concatenate(images)
    lm = np.concatenate(lm)
    num_img = images.shape[0]

    # Shuffle landmarks
    random_indices = rng.permutation(num_img)
    images = images[random_indices]
    lm = lm[random_indices]

    # Pad 0 to landmark labels
    lm = np.concatenate((np.ones((num_img, 1), dtype=np.float32), lm), axis=1)

    # Split landmarks into train, validation and test
    val_lm_img = images[:19000]
    val_lm_lm = lm[:19000]
    test_lm_img = images[19000:38113]
    test_lm_lm = lm[19000:38113]

    # Save landmarks into files
    for i in range(3):
        np.save(in_dir + '/../train/lm_img_' + str(i) + '.npy', images[38113 + i * 90000:38113 + (i + 1) * 90000])
        np.save(in_dir + '/../train/lm_lm_' + str(i) + '.npy', lm[38113 + i * 90000:38113 + (i + 1) * 90000])
    
    # Clear some variables
    del images
    del lm

    # Create validation set
    val_img = np.concatenate((val_pos_img, val_par_img, val_neg_img, val_lm_img))
    val_pos_class = np.ones((9500, 3), np.float32)
    val_pos_class[:, 2] = 0
    val_par_class = np.zeros((9500, 3), np.float32)
    val_neg_class = np.ones((9500, 3), np.float32)
    val_neg_class[:, 1] = 0
    val_lm_class = np.zeros((19000, 3), np.float32)
    val_class = np.concatenate((val_pos_class, val_par_class, val_neg_class, val_lm_class))
    val_neg_lm_bb = np.zeros((28500, 5), np.float32)
    val_bb = np.concatenate((val_pos_bb, val_par_bb, val_neg_lm_bb))
    val_pos_par_neg_lm = np.zeros((28500, 11), np.float32)
    val_lm = np.concatenate((val_pos_par_neg_lm, val_lm_lm))

    # Shuffle validation set
    random_indices = rng.permutation(47500)
    val_img = val_img[random_indices]
    val_class = val_class[random_indices]
    val_bb = val_bb[random_indices]
    val_lm = val_lm[random_indices]

    # Save validation set
    np.save(in_dir + '/../validation/images.npy', val_img)
    np.save(in_dir + '/../validation/class_labels.npy', val_class)
    np.save(in_dir + '/../validation/bounding_box_labels.npy', val_bb)
    np.save(in_dir + '/../validation/landmark_labels.npy', val_lm)

    # Create test set
    test_img = np.concatenate((test_pos_img, test_par_img, test_neg_img, test_lm_img))
    test_pos_class = np.ones((9500, 3), np.float32)
    test_pos_class[:, 2] = 0
    test_par_class = np.zeros((9500, 3), np.float32)
    test_neg_class = np.ones((9500, 3), np.float32)
    test_neg_class[:, 1] = 0
    test_lm_class = np.zeros((19113, 3), np.float32)
    test_class = np.concatenate((test_pos_class, test_par_class, test_neg_class, test_lm_class))
    test_neg_lm_bb = np.zeros((28613, 5), np.float32)
    test_bb = np.concatenate((test_pos_bb, test_par_bb, test_neg_lm_bb))
    test_pos_par_neg_lm = np.zeros((28500, 11), np.float32)
    test_lm = np.concatenate((test_pos_par_neg_lm, test_lm_lm))

    # Shuffle test set
    random_indices = rng.permutation(47613)
    test_img = test_img[random_indices]
    test_class = test_class[random_indices]
    test_bb = test_bb[random_indices]
    test_lm = test_lm[random_indices]

    # Save test set
    np.save(in_dir + '/../test/images.npy', test_img)
    np.save(in_dir + '/../test/class_labels.npy', test_class)
    np.save(in_dir + '/../test/bounding_box_labels.npy', test_bb)
    np.save(in_dir + '/../test/landmark_labels.npy', test_lm)

if __name__ == "__main__":
    main()