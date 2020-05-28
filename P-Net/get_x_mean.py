import argparse
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description='Get x_mean for zero centering the data since we forgot to do it when the data were being generated.')
    parser.add_argument(
        '--data-folder',
        type=str,
        help='The path to the folder that contains data',
        required=True
    )
    return parser.parse_args()

def main():
    args = get_arguments()
    x_train = np.load(args.data_folder + '/train/images.npy').astype(np.float32)
    x_train = x_train[:, :, :, ::-1] # Convert images from bgr to rgb
    x_train = x_train / 255 # Convert images from integer [0, 255] to float [0, 1]
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    np.save(args.data_folder + '/x_mean.npy', x_mean)

if __name__ == "__main__":
    main()