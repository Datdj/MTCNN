import sys
sys.path.insert(0, '..')
import argparse
import numpy as np
import tensorflow as tf
from utils.data_augmentation import augment
from tensorflow.data import Dataset
from model import pnet
from utils.losses import BCE_with_sti, MSE_with_sti
from utils.custom_metrics import accuracy_, recall_
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from os import path
import os

def main():

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
    args = parser.parse_args()
    models_dir = args.models_directory
    if args.log_frequency != 'epoch' and args.log_frequency != 'batch':
        log_frequency = int(args.log_frequency)
    else:
        log_frequency = args.log_frequency

    # Load training data
    x_train = np.load(args.data_folder + '/train/images.npy')
    y1_train = np.load(args.data_folder + '/train/class_labels.npy')
    y2_train = np.load(args.data_folder + '/train/bounding_box_labels.npy')
    y3_train = np.load(args.data_folder + '/train/landmark_labels.npy')

    # Create a data pipeline
    train_dataset = Dataset.from_tensor_slices((x_train, (y1_train, y2_train, y3_train)))
    train_dataset = train_dataset.shuffle(x_train.shape[0]).batch(args.batch_size, drop_remainder=True)
    train_dataset = train_dataset.map(augment)

    # Load validation data
    x_validation = np.load(args.data_folder + '/validation/images.npy')
    x_validation = np.add(x_validation, -127.5, dtype=np.float32) / 127.5
    y1_validation = np.load(args.data_folder + '/validation/class_labels.npy')
    y2_validation = np.load(args.data_folder + '/validation/bounding_box_labels.npy')
    y3_validation = np.load(args.data_folder + '/validation/landmark_labels.npy')

    # Create validation dataset
    validation_dataset = Dataset.from_tensor_slices((x_validation, (y1_validation, y2_validation, y3_validation))).batch(args.batch_size, drop_remainder=True)

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
    model = pnet()
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
        x=train_dataset,
        epochs=args.num_epochs,
        callbacks=[early_stopping, model_checkpoint, lr_decay, tensorboard],
        validation_data=validation_dataset
    )

if __name__ == "__main__":
    main()