import os

import pandas as pd
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import keras
from keras import optimizers

from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, concatenate

from keras.models import Model
from keras.callbacks import ModelCheckpoint

from classify import classify_for_dual_input_and_split_to_train_test


def create_convolution_layers(input_img, input_shape):
    model = Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu')(input_img)
    model = MaxPooling2D((2, 2), padding='same')(model)
    model = Dropout(0.25)(model)
    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2), padding='same')(model)
    model = Dropout(0.25)(model)
    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2), padding='same')(model)
    model = Dropout(0.4)(model)
    return model


def create_dual_cnn(input_shape, num_classes):
    current_input = Input(shape=input_shape)
    current_model = create_convolution_layers(current_input)

    voltage_input = Input(shape=input_shape)
    voltage_model = create_convolution_layers(voltage_input)

    conv = concatenate([current_model, voltage_model])

    conv = Flatten()(conv)

    dense = Dense(512)(conv, activation = 'relu')

    dense = Dropout(0.5)(dense)

    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[current_input, voltage_input], outputs=[output])

    opt = optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


if __name__ == '__main__':
    img_rows, img_cols = 100, 100
    as_gray = False
    in_channel = 4
    if as_gray:
        in_channel = 1
    num_classes = 11  # number of appliances
    batch_size = 32
    epochs = 100
    input_shape = (img_rows, img_cols, in_channel)
    input_img = Input(shape=input_shape)
    path = '/Users/gali.k/phd/phd_2021/stimuli/equate_1/images_1'
    x_train_right, x_train_left, labels = classify_for_dual_input_and_split_to_train_test(mode='count',
                                                                                       path=path,
                                                                                       stimuli_type='katzin', one_hot=True, as_gray=as_gray,
                                                                                       img_rows=img_rows, img_cols=img_cols, channels=in_channel, num_classes=11)


    x_train_comp = np.stack((x_train_right, x_train_left), axis=4)

    x_train, x_test, y_train, y_test = train_test_split(x_train_comp, labels, test_size=0.3, random_state=666)

    # take them apart
    x_train_right = x_train[:, :, :, :, 0]
    x_test_right = x_test[:, :, :, :, 0]

    x_train_left = x_train[:, :, :, :, 1]
    x_test_left = x_test[:, :, :, :, 1]

    best_weights_file = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(best_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    callbacks = [checkpoint]

    model = create_dual_cnn(input_shape, num_classes)
    model.fit([x_train_right, x_train_left], y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              verbose=1,
              validation_data=([x_test_right, x_test_left], y_test),
              shuffle=True)

    final_loss, final_acc = model.evaluate([x_test_right, x_test_left], y_test, verbose=1)
    print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))