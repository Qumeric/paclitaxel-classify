import mahotas as mh
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from params import img_rows, img_cols, train_split, image_dirs, nb_classes
from pathlib import Path


def img_to_float(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return (X_train, y_train), (X_test, y_test)


def load_data(augmented=True, image_dirs=image_dirs):
    X = []
    y = []

    suff = 'A' if augmented else ''

    for i in range(len(image_dirs)):
        for pth in (Path.cwd() / image_dirs[i] / ('data' + str(img_rows) + suff)).iterdir():
            im = mh.imread(str(pth), True)
            X.append(im)
            y.append(i)

    p = np.random.permutation(len(X))

    X = [X[i] for i in p]
    y = [y[i] for i in p]

    if train_split == 0:  # Use everything both to train and validate
        X_train = np.array(X, np.uint8)
        y_train = np.array(y, np.uint8)
        X_test = X_train[::]
        y_test = y_train[::]
    else:
        train_size = int(len(y) * train_split)
        X_train = np.array(X[:train_size], np.uint8)
        y_train = np.array(y[:train_size], np.uint8)
        X_test = np.array(X[train_size:], np.uint8)
        y_test = np.array(y[train_size:], np.uint8)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return img_to_float(X_train, y_train, X_test, y_test)


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return img_to_float(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_data()
    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
