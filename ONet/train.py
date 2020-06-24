import sys
sys.path.insert(0, '..')
import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from numpy.random import default_rng
from matplotlib import pyplot as plt
import matplotlib
import argparse
from utils.data_augmentation import augment_v2
from model import onet
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.losses import BCE_with_sti, MSE_with_sti
from utils.custom_metrics import accuracy_, recall_
from os import path
import os

def get_argument():
    parser = argparse.ArgumentParser(description='P-Net training.')
    parser.add_argument(
        '--data-folder',
        type=str,
        help='The path to the folder that contains data',
        required=True
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size',
        required=True
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate',
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        help='Number of epochs to train the model',
        required=True
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        help='After n epochs, if validation loss does not improve, stop training',
        required=True
    )
    parser.add_argument(
        '--lr-decay-patience',
        type=int,
        help='After n epochs, if validation loss does not improve, reduce the learning rate',
        required=True
    )
    parser.add_argument(
        '--lr-decay-min-delta',
        type=float,
        help='After some epochs, if validation loss does not decrease at least min_delta, reduce the learning rate'
    )
    parser.add_argument(
        '--models-directory',
        type=str,
        help='Where to save trained models',
        required=True
    )
    parser.add_argument(
        '--hard-sample-mining',
        type=int,
        help='hard-sample-mining=1 will use online hard sample training',
        required=True
    )
    parser.add_argument(
        '--num-back',
        type=int,
        help='Number of samples to backprop',
        required=True
    )
    parser.add_argument(
        '--log-frequency',
        type=str,
        help='The frequency at which to log. It can be "epoch" or "batch" or integer',
        required=True
    )
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        help='Steps per epoch',
        required=True
    )
    return parser.parse_args()

def onet_generator(batch_size, steps_per_epoch, data_folder):
    
    def generator():

        # Instantiate a random generator
        rng = default_rng()

        ppn_bs = round(batch_size / 5)
        lm_bs = batch_size - 3 * ppn_bs
        num = (steps_per_epoch // 3) * ppn_bs

        neg_lm_bb = np.zeros((ppn_bs + lm_bs, 5), np.float32)
        pos_par_neg_lm = np.zeros((3 * ppn_bs, 11), np.float32)
        batch_class = np.zeros((batch_size, 3), np.float32)
        batch_class[:ppn_bs, :2] = 1
        batch_class[2 * ppn_bs:3 * ppn_bs, ::2] = 1

        # Each iteration of this loop is an epoch
        while True:

            random_pos = rng.permutation(6)
            random_par = rng.permutation(3)
            random_neg = rng.permutation(3)
            random_lm = rng.permutation(3)

            # Each iteration of this loop is one third of an epoch
            for i in range(3):

                # Load positive data
                pos_img = list()
                pos_bb = list()
                for j in range(2):
                    img = np.load(data_folder + '/train/pos_img_' + str(random_pos[2 * i + j]) + '.npy')
                    anno = np.load(data_folder + '/train/pos_bb_' + str(random_pos[2 * i + j]) + '.npy')
                    num_img = img.shape[0]
                    random_indices = rng.permutation(num_img)
                    img = img[random_indices]
                    anno = anno[random_indices]
                    pos_img.append(img[:num // 2])
                    pos_bb.append(anno[:num // 2])
                    del img
                pos_img = np.concatenate(pos_img)
                pos_bb = np.concatenate(pos_bb)

                # Load part face data
                par_img = np.load(data_folder + '/train/par_img_' + str(random_par[i]) + '.npy')
                par_bb = np.load(data_folder + '/train/par_bb_' + str(random_par[i]) + '.npy')
                num_img = par_img.shape[0]
                random_indices = rng.permutation(num_img)
                par_img = par_img[random_indices][:num]
                par_bb = par_bb[random_indices][:num]

                # Load negative data
                neg_img = np.load(data_folder + '/train/neg_img_' + str(random_neg[i]) + '.npy')
                num_img = neg_img.shape[0]
                random_indices = rng.permutation(num_img)
                neg_img = neg_img[random_indices][:num]

                # Load landmark data
                lm_img = np.load(data_folder + '/train/lm_img_' + str(random_lm[i]) + '.npy')
                lm_lm = np.load(data_folder + '/train/lm_lm_' + str(random_lm[i]) + '.npy')
                num_img = lm_img.shape[0]
                random_indices = rng.permutation(num_img)
                lm_img = lm_img[random_indices]
                lm_lm = lm_lm[random_indices]

                # Each iteration of this loop is a batch
                for j in range(steps_per_epoch // 3):
                    batch_img = np.concatenate((
                        pos_img[j * ppn_bs:(j + 1) * ppn_bs],
                        par_img[j * ppn_bs:(j + 1) * ppn_bs],
                        neg_img[j * ppn_bs:(j + 1) * ppn_bs],
                        lm_img[j * lm_bs:(j + 1) * lm_bs]
                    ))
                    batch_bb = np.concatenate((
                        pos_bb[j * ppn_bs:(j + 1) * ppn_bs],
                        par_bb[j * ppn_bs:(j + 1) * ppn_bs],
                        neg_lm_bb
                    ))
                    batch_lm = np.concatenate((
                        pos_par_neg_lm,
                        lm_lm[j * lm_bs:(j + 1) * lm_bs]
                    ))
                    yield batch_img, (batch_class, batch_bb, batch_lm)

    return generator

def load_validation_data(data_folder, batch_size):
    x_validation = np.load(data_folder + '/validation/images.npy')
    x_validation = np.add(x_validation, -127.5, dtype=np.float32) / 127.5
    y1_validation = np.load(data_folder + '/validation/class_labels.npy')
    y2_validation = np.load(data_folder + '/validation/bounding_box_labels.npy')
    y3_validation = np.load(data_folder + '/validation/landmark_labels.npy')
    return Dataset.from_tensor_slices((x_validation, (y1_validation, y2_validation, y3_validation))).batch(batch_size, drop_remainder=True)

def main():

    # Get arguments
    args = get_argument()
    models_dir = args.models_directory
    if args.log_frequency != 'epoch' and args.log_frequency != 'batch':
        log_frequency = int(args.log_frequency)
    else:
        log_frequency = args.log_frequency

    # Prepare training dataset
    train_data = Dataset.from_generator(
        generator=onet_generator(args.batch_size, args.steps_per_epoch, args.data_folder),
        output_types=(tf.uint8, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(tf.TensorShape([args.batch_size, 48, 48, 3]), (tf.TensorShape([args.batch_size, 3]), tf.TensorShape([args.batch_size, 5]), tf.TensorShape([args.batch_size, 11])))
    )
    train_data = train_data.map(augment_v2)

    # Prepare validation dataset
    val_data = load_validation_data(args.data_folder, args.batch_size)

    # Stop training if no improvements are made
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping,
        mode='min'
    )

    # Model checkpoints
    model_checkpoint = ModelCheckpoint(
        filepath=models_dir + '/epoch_{epoch:04d}_val_loss_{val_loss:.4f}.hdf5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )

    # Learning rate decay
    lr_decay = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=args.lr_decay_patience,
        mode='min',
        min_delta=args.lr_decay_min_delta
    )

    # Set up Tensorboard
    tensorboard = TensorBoard(
        log_dir=models_dir + '/log',
        write_graph=False,
        profile_batch=0,
        update_freq=log_frequency
    )

    # Create and compile the model from scratch
    model = onet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=[
            BCE_with_sti(args.hard_sample_mining, args.num_back),
            MSE_with_sti(args.hard_sample_mining, args.num_back),
            MSE_with_sti(args.hard_sample_mining, args.num_back)
        ],
        metrics=[[accuracy_(), recall_()], None, None],
        loss_weights=[1, 0.5, 1]
    )

    # Create folders
    if not path.exists(models_dir):
        os.makedirs(models_dir)
    if not path.exists(models_dir + '/log'):
        os.mkdir(models_dir + '/log')

    # Train the model
    history = model.fit(
        x=train_data,
        epochs=args.num_epochs,
        callbacks=[early_stopping, model_checkpoint, lr_decay, tensorboard],
        validation_data=val_data,
        steps_per_epoch=args.steps_per_epoch
    )

if __name__ == "__main__":
    main()