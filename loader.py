import mahotas as mh
import numpy as np
from pathlib import Path
from params import img_size, train_split


def load_data():
    image_dirs= ['01mkg', '1mkg', '5mkg', 'control']
    #image_dirs= ['5mkg', 'control']
    #image_dirs = ['01mkg', '1mkg']
    #image_dirs = ['control', '1mkg']
    #image_dirs = ['01mkg', '1mkg']
    X = []
    y = []

    for i in range(len(image_dirs)):
        for pth in (Path.cwd() / 'images' / image_dirs[i] / ('data'+str(img_size))).iterdir():
            im = mh.imread(str(pth), True)
            X.append(im)
            y.append(i)
    
    p = np.random.permutation(len(X))

    X = [X[i] for i in p]
    y = [y[i] for i in p]

    if train_split == 0:
        X_train = np.array(X, np.uint8)
        y_train = np.array(y, np.uint8)
        X_test = X_train[::]
        y_test = y_train[::]
    else:
        train_size = int(len(y)*train_split)
        X_train = np.array(X[:train_size], np.uint8)
        y_train = np.array(y[:train_size], np.uint8)
        X_test = np.array(X[train_size:], np.uint8)
        y_test = np.array(y[train_size:], np.uint8)         
    
    img_rows = X[0].shape[0]
    img_cols = X[0].shape[1]
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
         
    return (X_train, y_train), (X_test, y_test)
    
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_data()
    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)