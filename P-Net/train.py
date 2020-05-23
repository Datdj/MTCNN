import numpy as np
import tensorflow as tf
from utils.data_augmentation import augment_and_zero_center
from tensorflow.data import Dataset
from model import pnet
from utils.losses import BCE_with_sample_type_indicator, MSE_with_sample_type_indicator
from utils.custom_metrics import accuracy_, recall_
import datetime

def main():

    parser = argparse.ArgumentParser(description='P-Net training.')
    parser.add_argument(
        '-d',
        '--data-folder',
        type=str,
        help='The path to the folder that contains data',
        required=True
    )
    parser.add_argument(
        '-bs',
        '--batch-size',
        type=int,
        help='Batch size',
        required=True
    )
    parser.add_argument(
        '-e',
        '--num-epochs',
        help='Number of epochs to train the model',
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

    # Set up Tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Load and compile the model
    model = pnet()
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
    model.fit(
        x=train_dataset,
        epochs=num_epochs,
        callbacks=[tensorboard_callback],
        validation_data=validation_dataset
    )

if __name__ == "__main__":
    main()