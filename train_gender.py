from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import *
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np

import matplotlib.pyplot as plt


GENDER_DATA_DIR = "<path to the dataset>"
SAVE_DIR = "<path where model weights have to be saved>"

POOL_SIZE = (2, 2)
LEARNING_RATE = 1e-3
DECAY = 1e-5
BATCH_SIZE = 128
EPOCHS = 30
TEST_SIZE = 0.20


def load_data(X, y):
    """
    This function loads the data directly into memory, since it fits in the memory. For a larger subset of IMDB
    dataset, Keras' ImageDataGenerator can be used to load images in smaller batches.
    :param X: (list) stores all images
    :param y: (list) stores all labels
    :return None:
    """
    print('Loading Female dataset...')
    for image in os.listdir(GENDER_DATA_DIR + "/" + "female"):
        img = cv2.imread(GENDER_DATA_DIR + "/" + "female/" + image, 0)
        img = cv2.resize(img, (128, 128))
        X.append(img)
        y.append(0)

    print('Loading Male dataset...')
    for image in os.listdir(GENDER_DATA_DIR + "/" + "male"):
        img = cv2.imread(GENDER_DATA_DIR + "/" + "male/" + image, 0)
        img = cv2.resize(img, (128, 128))
        X.append(img)
        y.append(1)


def create_model(input_shape):
    """
    This function adds layers to the sequential model and returns it after compiling.

    :param input_shape: (int, int) input shape of the batches of images that pass through the network
    :return model: (keras.engine.sequential.Sequential) compiled cnn model
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(Adam(lr=LEARNING_RATE, decay=DECAY), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def plot_training_results(history):
    """
    This function plots accuracy and loss using the History object returned by fitting the model.

    :param history: (History) History object
    :return None:
    """
    # plotting our results
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    X = []
    y = []
    load_data(X, y)
    X, y = np.array(X)/255.0, np.array(y)

    # required dimensions for cnn model
    X = X.reshape(X.shape[0], 128, 128, 1)
    # convert to categorical format
    y = to_categorical(y, 2)
    # shuffle image-label pairs
    X, y = shuffle(X, y)
    # train-test split 80%-20%
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE)

    print("TRAINING_DATA SHAPE: ", x_train.shape)
    print("VALIDATION_DATA SHAPE: ", x_val.shape)

    model_input_shape = x_train.shape[1:]
    cnn = create_model(model_input_shape)
    # shows model summary
    cnn.summary()
    # here we save the weights when we observe minimum validation loss
    filepath = SAVE_DIR + "/weights-{epoch:02d}-{val_loss:4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # fit the model
    history = cnn.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, verbose=2,
                      callbacks=[checkpoint])

    # plot the results
    plot_training_results(history)
