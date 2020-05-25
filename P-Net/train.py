import numpy as np
import tensorflow as tf
from utils.data_augmentation import augment_and_zero_center
from tensorflow.data import Dataset
from model import pnet
from utils.losses import BCE_with_sample_type_indicator, MSE_with_sample_type_indicator
from utils.custom_metrics import accuracy_, recall_
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dateutil import tz
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
    args = parser.parse_args()

    # Load training data
    x_train = np.load(args.data_folder + '/train/images.npy').astype(np.float32)
    num_train = x_train.shape[0]
    x_train = x_train[:, :, :, ::-1] # Convert images from bgr to rgb
    x_train = x_train / 255 # Convert images from integer [0, 255] to float [0, 1)
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
    x_validation = np.load(args.data_folder + './validation/images.npy')
    x_validation = x_validation[:, :, :, ::-1] # Convert images from bgr to rgb
    x_validation = x_validation / 255 # Convert images from integer [0, 255] to float [0, 1)
    x_validation = x_validation - mean_x_train # Zero center the images
    y1_validation = np.load(args.data_folder + './validation/class_labels.npy').astype(np.float32)
    y2_validation = np.load(args.data_folder + './validation/bounding_box_labels.npy').astype(np.float32)
    y3_validation = np.load(args.data_folder + './validation/landmark_labels.npy').astype(np.float32)

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
        filepath=args.models_directory,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        save_freq='epoch'
    )

    # Learning rate decay
    lr_decay = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=args.lr_decay,
        mode='min'
    )

    # Set up Tensorboard
    if not path.exists('./log'):
        os.mkdir('./log')
    tensorboard = TensorBoard(
        log_dir="./log/" + datetime.now(tz.gettz('UTC+7')).strftime("%Y-%m-%d_%H-%M-%S-%p"),
        write_graph=False,
        profile_batch=0
    )

    # Load and compile the model
    model = pnet(batch_size=args.batch_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[
            BCE_with_sample_type_indicator,
            MSE_with_sample_type_indicator,
            MSE_with_sample_type_indicator
        ],
        metrics=[[accuracy_(), recall_()], None, None]
        loss_weights=[1, 0.5, 0.5]
    )

    # Train the model
    history = model.fit(
        x=train_dataset,
        epochs=num_epochs,
        callbacks=[early_stopping, model_checkpoint, lr_decay, tensorboard],
        validation_data=validation_dataset
    )

if __name__ == "__main__":
    main()