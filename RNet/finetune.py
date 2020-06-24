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
from model import rnet
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
    parser.add_argument(
        '--rnet',
        type=str,
        help='The path to the .hdf5 file containing the weights of R-Net',
        required=True
    )
    return parser.parse_args()

def rnet_generator(batch_size, steps_per_epoch, data_folder):
    def generator():
        
        rng = default_rng()

        pos_par_neg_batch_size = round(batch_size / 5)
        lm_batch_size = batch_size - pos_par_neg_batch_size * 3
        num_pos_par_neg = pos_par_neg_batch_size * steps_per_epoch
        num_lm = lm_batch_size * steps_per_epoch

        pos_class = np.ones((pos_par_neg_batch_size, 3), np.float32)
        pos_class[:, 2] = 0
        
        par_class = np.zeros((pos_par_neg_batch_size, 3), np.float32)
        
        neg_class = np.ones((pos_par_neg_batch_size, 3), np.float32)
        neg_class[:, 1] = 0
        neg_bb = np.zeros((pos_par_neg_batch_size, 5), np.float32)

        lm_class = np.zeros((lm_batch_size, 3), np.float32)
        lm_bb = np.zeros((lm_batch_size, 5), np.float32)

        pos_par_neg_lm = np.zeros((pos_par_neg_batch_size * 3, 11), dtype=np.float32)

        batch_class = np.concatenate((pos_class, par_class, neg_class, lm_class))

        # Each iteration of this loop is an epoch
        while True:
            
            # Load positive data
            img = np.load(data_folder + '/train/pos_img.npy')
            pos_bb = np.load(data_folder + '/train/pos_bb.npy')
            # print(img.shape)
            random_indices = rng.permutation(img.shape[0])
            pos_img = img[random_indices][:num_pos_par_neg]
            # print(pos_img.shape)
            pos_bb = pos_bb[random_indices][:num_pos_par_neg]
            del img

            # Load part face data
            img = np.load(data_folder + '/train/par_img.npy')
            par_bb = np.load(data_folder + '/train/par_bb.npy')
            # print(img.shape)
            random_indices = rng.permutation(img.shape[0])
            par_img = img[random_indices][:num_pos_par_neg]
            # print(par_img.shape)
            par_bb = par_bb[random_indices][:num_pos_par_neg]
            del img

            # Load negative data
            img = np.load(data_folder + '/train/neg_img.npy')
            # print(img.shape)
            random_indices = rng.permutation(img.shape[0])
            neg_img = img[random_indices][:num_pos_par_neg]
            # print(neg_img.shape)
            del img

            # Load lanmark data
            img = np.load(data_folder + '/train/lm_img.npy')
            lm_lm = np.load(data_folder + '/train/lm_lm.npy')
            # print(img.shape)
            random_indices = rng.permutation(img.shape[0])
            lm_img = img[random_indices][:num_lm]
            # print(lm_img.shape)
            lm_lm = lm_lm[random_indices][:num_lm]
            del img

            # Each iteration of this loop is a batch
            for i in range(steps_per_epoch):
                batch_img = np.concatenate((
                    pos_img[i * pos_par_neg_batch_size:(i + 1) * pos_par_neg_batch_size],
                    par_img[i * pos_par_neg_batch_size:(i + 1) * pos_par_neg_batch_size],
                    neg_img[i * pos_par_neg_batch_size:(i + 1) * pos_par_neg_batch_size],
                    lm_img[i * lm_batch_size:(i + 1) * lm_batch_size]
                ))
                batch_bb = np.concatenate((
                    pos_bb[i * pos_par_neg_batch_size:(i + 1) * pos_par_neg_batch_size],
                    par_bb[i * pos_par_neg_batch_size:(i + 1) * pos_par_neg_batch_size],
                    neg_bb,
                    lm_bb
                ))
                batch_lm = np.concatenate((
                    pos_par_neg_lm,
                    lm_lm[i * lm_batch_size:(i + 1) * lm_batch_size]
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

    # a = rnet_generator(args.batch_size, args.steps_per_epoch, args.data_folder)()
    # for i in range(15723):
    #     if next(a)[0].shape[0] != 32:
    #         print(i)
    #         break
    # exit()

    # Prepare training dataset
    train_data = Dataset.from_generator(
        generator=rnet_generator(args.batch_size, args.steps_per_epoch, args.data_folder),
        output_types=(tf.uint8, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(tf.TensorShape([args.batch_size, 24, 24, 3]), (tf.TensorShape([args.batch_size, 3]), tf.TensorShape([args.batch_size, 5]), tf.TensorShape([args.batch_size, 11])))
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
    model = rnet()

    # Load the pre-trained model for finetuning
    model.load_weights(filepath=args.rnet, by_name=True)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=[
            BCE_with_sti(args.hard_sample_mining, args.num_back),
            MSE_with_sti(args.hard_sample_mining, args.num_back),
            MSE_with_sti(args.hard_sample_mining, args.num_back)
        ],
        metrics=[[accuracy_(), recall_()], None, None],
        loss_weights=[1, 0.5, 0.5]
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