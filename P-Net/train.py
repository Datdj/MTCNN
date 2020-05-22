import numpy as np
import tensorflow as tf
from utils.data_augmentation import augment_and_zero_center
from tensorflow.data import Dataset
from model import pnet
from utils.losses import BCE_with_sample_type_indicator, MSE_with_sample_type_indicator

def main():

    parser = argparse.ArgumentParser(description='P-Net training.')
    parser.add_argument(
        '-d',
        '--data-folder',
        type=str,
        help='The path to the folder that contains data',
        required=True
    )
    args = parser.parse_args()

    # Load data
    x_train = np.load(args.data_folder + '/train/images.npy')
    x_train = x_train[:, :, :, ::-1] # Convert images from bgr to rgb
    x_train = x_train / 255 # Convert images from integer [0, 255] to float [0, 1)
    mean_x_train = tf.math.reduce_mean(x_train, axis=0, keepdims=True) # Compute the mean image to make the images zero centered (kind of) later
    y1_train = np.load(args.data_folder + '/train/class_labels.npy')
    y2_train = np.load(args.data_folder + '/train/bounding_box_labels.npy')
    y3_train = np.load(args.data_folder + '/train/landmark_labels.npy')
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, (y1_train, y2_train, y3_train)))

    # Create data pipeline
    train_dataset = Dataset.from_tensor_slices((x_train, (y1_train, y2_train, y3_train)))
    train_dataset = train_dataset.shuffle(496883).batch(2, drop_remainder=True).repeat()
    train_dataset = train_dataset.map(augment_and_zero_center(mean=mean_x_train))

    # Load and compile the model
    model = pnet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[
            BCE_with_sample_type_indicator,
            MSE_with_sample_type_indicator,
            MSE_with_sample_type_indicator
        ],
        metrics=[[accuracy(), recall()], [], []],
        loss_weights=[1, 0.5, 0.5]
    )    

if __name__ == "__main__":
    main()