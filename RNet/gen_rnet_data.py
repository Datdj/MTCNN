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

    pos_img = list()
    par_img = list()
    neg_img = list()
    pos_annotation = list()
    par_annotation = list()

    pos_img.append(np.load(in_dir + '/pos_images_rnet.npy'))
    pos_annotation.append(np.load(in_dir + '/positives_rnet.npy'))
    lm_img = np.load(in_dir + '/lm_images.npy')
    lm = np.load(in_dir + '/landmarks.npy')

    for i in range(13):
        pos_img.append(np.load(in_dir + '/positive_images_' + str(i) + '.npy'))
        par_img.append(np.load(in_dir + '/part_face_images' + str(i) + '.npy'))
        neg_img.append(np.load(in_dir + '/negative_images' + str(i) + '.npy'))
        pos_annotation.append(np.load(in_dir + '/positives' + str(i) + '.npy'))
        par_annotation.append(np.load(in_dir + '/part_faces' + str(i) + '.npy'))

    pos_img = np.concatenate(pos_img)
    # pos_img = pos_img[0]
    par_img = np.concatenate(par_img)
    neg_img = np.concatenate(neg_img)
    pos_annotation = np.concatenate(pos_annotation)
    # pos_annotation = pos_annotation[0]
    par_annotation = np.concatenate(par_annotation)
    # print(pos_img.shape)
    # print(pos_annotation.shape)
    # exit()

    num_pos = pos_img.shape[0]
    num_par = par_img.shape[0]
    num_neg = neg_img.shape[0]
    num_lm = lm_img.shape[0]

    # Create classification labels
    pos_class = np.concatenate((np.ones((num_pos, 2), dtype=np.float32), np.zeros((num_pos, 1), dtype=np.float32)), axis=1)
    par_class = np.zeros((num_par, 3), dtype=np.float32)
    neg_class = np.concatenate((np.ones((num_neg, 1), dtype=np.float32), np.zeros((num_neg, 1), dtype=np.float32), np.ones((num_neg, 1), dtype=np.float32)), axis=1)
    lm_class = np.zeros((num_lm, 3), dtype=np.float32)

    # Create bounding box regression labels
    pos_bb = np.concatenate((np.ones((num_pos, 1), dtype=np.float32), pos_annotation), axis=1)
    par_bb = np.concatenate((np.ones((num_par, 1), dtype=np.float32), par_annotation), axis=1)
    neg_bb = np.zeros((num_neg, 5), dtype=np.float32)
    lm_bb = np.zeros((num_lm, 5), dtype=np.float32)

    # Create landmark regression labels
    pos_lm = np.zeros((num_pos, 11), dtype=np.float32)
    par_lm = np.zeros((num_par, 11), dtype=np.float32)
    neg_lm = np.zeros((num_neg, 11), dtype=np.float32)
    lm_lm = np.concatenate((np.ones((num_lm, 1), dtype=np.float32), lm), axis=1)

    # Shuffle positives
    random_indices = rng.permutation(num_pos)
    pos_img = pos_img[random_indices]
    pos_bb = pos_bb[random_indices]

    # Shuffle part faces
    random_indices = rng.permutation(num_par)
    par_img = par_img[random_indices]
    par_bb = par_bb[random_indices]

    # Shuffle negatives
    random_indices = rng.permutation(num_neg)
    neg_img = neg_img[random_indices]

    # Shuffle landmarks
    random_indices = rng.permutation(num_lm)
    lm_img = lm_img[random_indices]
    lm_lm = lm_lm[random_indices]

    # Create validation set
    val_img = np.concatenate((pos_img[:10000], par_img[:10000], neg_img[:10000], lm_img[:20000]))
    # print(val_img.shape)
    # exit()
    val_class = np.concatenate((pos_class[:10000], par_class[:10000], neg_class[:10000], lm_class[:20000]))
    val_bb = np.concatenate((pos_bb[:10000], par_bb[:10000], neg_bb[:10000], lm_bb[:20000]))
    val_lm = np.concatenate((pos_lm[:10000], par_lm[:10000], neg_lm[:10000], lm_lm[:20000]))
    random_indices = rng.permutation(50000)
    val_img = val_img[random_indices]
    val_class = val_class[random_indices]
    val_bb = val_bb[random_indices]
    val_lm = val_lm[random_indices]

    # Create test set
    test_img = np.concatenate((pos_img[10000:20000], par_img[10000:20000], neg_img[10000:20000], lm_img[20000:40000]))
    test_class = np.concatenate((pos_class[10000:20000], par_class[10000:20000], neg_class[10000:20000], lm_class[20000:40000]))
    test_bb = np.concatenate((pos_bb[10000:20000], par_bb[10000:20000], neg_bb[10000:20000], lm_bb[20000:40000]))
    test_lm = np.concatenate((pos_lm[10000:20000], par_lm[10000:20000], neg_lm[10000:20000], lm_lm[20000:40000]))
    random_indices = rng.permutation(50000)
    test_img = test_img[random_indices]
    test_class = test_class[random_indices]
    test_bb = test_bb[random_indices]
    test_lm = test_lm[random_indices]

    # Save training data
    np.save(in_dir + '/../train/pos_img.npy', pos_img[20000:])
    np.save(in_dir + '/../train/pos_bb.npy', pos_bb[20000:])
    np.save(in_dir + '/../train/par_img.npy', par_img[20000:])
    np.save(in_dir + '/../train/par_bb.npy', par_bb[20000:])
    np.save(in_dir + '/../train/neg_img.npy', neg_img[20000:])
    np.save(in_dir + '/../train/lm_img.npy', lm_img[40000:])
    np.save(in_dir + '/../train/lm_lm.npy', lm_lm[40000:])

    # Save validation and test data
    np.save(in_dir + '/../validation/images.npy', val_img)
    np.save(in_dir + '/../validation/class_labels.npy', val_class)
    np.save(in_dir + '/../validation/bounding_box_labels.npy', val_bb)
    np.save(in_dir + '/../validation/landmark_labels.npy', val_lm)
    np.save(in_dir + '/../test/images.npy', test_img)
    np.save(in_dir + '/../test/class_labels.npy', test_class)
    np.save(in_dir + '/../test/bounding_box_labels.npy', test_bb)
    np.save(in_dir + '/../test/landmark_labels.npy', test_lm)

    # print('Positives:')
    # print(pos_img.shape)
    # print(pos_annotation.shape)
    # print(pos_class.shape)
    # print(pos_bb.shape)
    # print(pos_lm.shape)
    # print('Part faces:')
    # print(par_img.shape)
    # print(par_annotation.shape)
    # print(par_class.shape)
    # print(par_bb.shape)
    # print(par_lm.shape)
    # print('Negatives:')
    # print(neg_img.shape)
    # print(neg_class.shape)
    # print(neg_bb.shape)
    # print(neg_lm.shape)
    # print('Landmarks:')
    # print(lm_img.shape)
    # print(lm.shape)
    # print(lm_class.shape)
    # print(lm_bb.shape)
    # print(lm_lm.shape)

if __name__ == "__main__":
    main()