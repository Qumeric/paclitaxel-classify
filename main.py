import matplotlib.pyplot as plt
from numpy.random import seed
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from loader import load_data
from params import img_rows, img_cols, nb_classes

(X_train, Y_train), (X_test, Y_test) = load_data()

seed(1337)

def get_model():
    model = Sequential()
    
    model.add(Convolution2D(32, 5, 5, input_shape=(1, img_rows, img_cols),
                            activation='relu', init='he_normal'))
    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


if __name__ == "__main__":
    nb_epoch=10
    batch_size=40
    model = get_model()
    
    hist = model.fit(X_train, Y_train, batch_size=40, nb_epoch=100, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    plt.plot(hist.history['val_acc'])
