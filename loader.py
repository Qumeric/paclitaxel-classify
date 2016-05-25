import mahotas as mh
import numpy as np
from pathlib import Path


def load_data(validation_split=0): # FIXME
    image_dirs= ['01mkg', '1mkg', '5mkg', 'control']
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(image_dirs)):
        for pth in (Path.cwd() / 'images' / image_dirs[i] / 'data').iterdir():
            im = mh.imread(str(pth), True)
            X_train.append(im)
            y_train.append(i)

    X_train = np.array(X_train, np.uint8)
    y_train = np.array(y_train, np.uint8)
           
    X_test = X_train[::] # FIXME
    y_test = y_train[::] # FIXME
    


         
    return (X_train, y_train), (X_test, y_test)
    
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_data()
    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)