import sys
sys.path.insert(0, '..')
import argparse
import numpy as np
import tensorflow as tf
from utils.data_augmentation import augment_and_zero_center
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
        '--lr-decay',
        type=int,
        help='After n epochs, if validation loss does not improve, reduce learning rate'
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
        '--initial-epoch',
        type=int,
        help='Use this when resuming training',
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Use this when resuming training',
        default=None
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
    x_train = np.load(args.data_folder + '/train/images.npy').astype(np.float32)
    num_train = x_train.shape[0]
    x_train = x_train[:, :, :, ::-1] # Convert images from bgr to rgb
    x_train = x_train / 255 # Convert images from integer [0, 255] to float [0, 1]
    mean_x_train = tf.math.reduce_mean(x_train, axis=0, keepdims=True) # Compute the mean image to make the images zero centered (kind of) later
    y1_train = np.load(args.data_folder + '/train/class_labels.npy').astype(np.float32)
    y2_train = np.load(args.data_folder + '/train/bounding_box_labels.npy').astype(np.float32)
    y3_train = np.load(args.data_folder + '/train/landmark_labels.npy').astype(np.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, (y1_train, y2_train, y3_train)))

    # Create data pipeline
    train_dataset = Dataset.from_tensor_slices((x_train, (y1_train, y2_train, y3_train)))
    train_dataset = train_dataset.shuffle(num_train).batch(args.batch_size, drop_remainder=True)
    train_dataset = train_dataset.map(augment_and_zero_center(mean=mean_x_train))

    # Load validation data
    x_validation = np.load(args.data_folder + '/validation/images.npy')
    x_validation = x_validation[:, :, :, ::-1] # Convert images from bgr to rgb
    x_validation = x_validation / 255 # Convert images from integer [0, 255] to float [0, 1]
    x_validation = x_validation - mean_x_train # Zero center the images
    y1_validation = np.load(args.data_folder + '/validation/class_labels.npy').astype(np.float32)
    y2_validation = np.load(args.data_folder + '/validation/bounding_box_labels.npy').astype(np.float32)
    y3_validation = np.load(args.data_folder + '/validation/landmark_labels.npy').astype(np.float32)

    # Create validation dataset
    validation_dataset = Dataset.from_tensor_slices((x_validation, (y1_validation, y2_validation, y3_validation))).batch(args.batch_size, drop_remainder=True)

    # Stop training if no improvements are made
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping,
        mode='min'
    )

    # Model checkpoints
    if not path.exists(models_dir):
        os.makedirs(models_dir)
    model_checkpoint = ModelCheckpoint(
        filepath=models_dir + '/epoch_{epoch:04d}_val_loss_{val_loss:.4f}.hdf5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    # Learning rate decay
    lr_decay = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=args.lr_decay,
        mode='min',
        min_delta=0.1
    )

    # Set up Tensorboard
    if not path.exists(models_dir + '/log'):
        os.mkdir(models_dir + '/log')
    tensorboard = TensorBoard(
        log_dir=models_dir + '/log',
        write_graph=False,
        profile_batch=0,
        update_freq=log_frequency
    )

    # Check whether we are resuming training or not
    if args.initial_epoch > 0:
        # Pick up where we left off
        if args.hard_sample_mining == 1:
            loss1_name = 'BCE_with_sti_and_hsm'
            loss2_name = 'MSE_with_sti_and_hsm'
        else:
            loss1_name = 'BCE_with_sample_type_indicator'
            loss2_name = 'MSE_with_sample_type_indicator'
        model = tf.keras.models.load_model(
            filepath=models_dir + '/' + args.model,
            custom_objects={
                loss1_name: BCE_with_sti(args.hard_sample_mining, args.num_back),
                loss2_name: MSE_with_sti(args.hard_sample_mining, args.num_back),
                '_accuracy': accuracy_(),
                'recall': recall_()
            }
        )
    else:
        # Create and compile the model from scratch
        model = pnet(batch_size=args.batch_size)
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

    # Train the model
    history = model.fit(
        x=train_dataset,
        epochs=args.num_epochs,
        callbacks=[early_stopping, model_checkpoint, lr_decay, tensorboard],
        validation_data=validation_dataset,
        initial_epoch=args.initial_epoch
    )

if __name__ == "__main__":
    main()