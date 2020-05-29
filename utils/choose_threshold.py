import sys
sys.path.insert(0, '..')
import argparse
import tensorflow as tf
from tensorflow.data import Dataset
from losses import BCE_with_sti, MSE_with_sti
from custom_metrics import accuracy_, recall_
import numpy as np
from matplotlib import pyplot as plt

def get_arguments():
    parser = argparse.ArgumentParser(description='Choose the best threshold for P-Net')
    parser.add_argument(
        '--data-folder',
        type=str,
        help='The path to the folder that contains data',
        required=True
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='The path to the trained model',
        required=True
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size',
        required=True
    )
    parser.add_argument(
        '--x-mean',
        type=str,
        help='The file containing x_mean',
        required=True
    )
    return parser.parse_args()

def load_val_data(data_folder, x_mean, batch_size):
    x_validation = np.load(data_folder + '/validation/images.npy')
    x_validation = x_validation[:, :, :, ::-1] # Convert images from bgr to rgb
    x_validation = x_validation / 255 # Convert images from integer [0, 255] to float [0, 1]
    x_validation = x_validation - x_mean # Zero center the images
    y1_validation = np.load(data_folder + '/validation/class_labels.npy').astype(np.float32)
    y2_validation = np.load(data_folder + '/validation/bounding_box_labels.npy').astype(np.float32)
    y3_validation = np.load(data_folder + '/validation/landmark_labels.npy').astype(np.float32)
    return Dataset.from_tensor_slices((x_validation, (y1_validation, y2_validation, y3_validation))).batch(batch_size, drop_remainder=True)

def evaluate(model, validation_data, batch_size):
    accs = np.zeros(101, dtype=np.float32)
    recalls = np.zeros(101, dtype=np.float32)
    for i in range(101):
        threshold = i / 100
        model.compile(metrics=[[accuracy_(threshold), recall_(threshold)], None, None])
        test = model.evaluate(x=validation_data, batch_size=batch_size)
        accs[i] = test[1]
        recalls[i] = test[2]
    return accs, recalls

def plot(accs, recalls):
    thresholds = np.arange(0, 101, dtype=np.float32) / 100
    plt.plot(thresholds, accs, 'g')
    plt.plot(thresholds, recalls, 'r')
    plt.show()

def main():
    # Get all the arguments
    args = get_arguments()
    # Load the model
    model = tf.keras.models.load_model(
        filepath=args.model_path,
        custom_objects={'tf': tf},
        compile=False
    )
    # Load the validation dataset
    val_data = load_val_data(args.data_folder, np.load(args.x_mean), args.batch_size)
    # Evaluate the model
    accs, recalls = evaluate(model, val_data, args.batch_size)
    # Plot the resuts
    plot(accs, recalls)

if __name__ == "__main__":
    main()