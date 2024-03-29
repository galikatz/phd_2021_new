import visualkeras
import os
import tensorflow as tf
import keras
from load_networks_and_test import load_model
import matplotlib as plt
import numpy as np
from keras.layers import Input, Dense, Dropout, Flatten

from keras.utils.vis_utils import plot_model


def plot_sequential_model():
    "/Users/gali.k/phd/phd_2021/visulization/simulation_5_balanced/equate_2/size/gens/model_2023-02-03_20_mode_size_equate_2_gen_15_individual_295_acc_0.908_loss_0.475_layers_3_neurons_[32, 16, 64]_activation_elu_optimizer_adam.h5"
    name = "model_2023-02-03_20_mode_size_equate_2_gen_15_individual_295_acc_0.908_loss_0.475_layers_3_neurons_[32, 16, 64]_activation_elu_optimizer_adam"
    PATH_FROM = f"/Users/gali.k/phd/phd_2021/visulization/simulation_5_balanced/equate_2/size/gens/"
    PATH_TO = f"/Users/gali.k/phd/phd_2021/visulization/simulation_5_balanced/equate_2/size/viz/"

    import sys;
    print('Python %s on %s' % (sys.version, sys.platform))
    sys.path.extend(['/Users/gali.k/phd/phd_2021'])
    model = load_model(PATH_FROM + name + ".h5")

    visualkeras.layered_view(model, legend=True, to_file=f'{PATH_TO}/{name}.png')  # write and show
    plot_model(model,to_file=f'{PATH_TO}/summary_{name}.png',show_shapes=True, show_layer_names=True)


def plot_multi_task_model():
    nb_layers = 3
    nb_neurons = [16, 32, 64]
    activation = 'tanh'
    optimizer = 'adamax'
    nb_classes = 2


    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
    from keras.models import Model

    inputs = Input(shape=(100, 100, 1))

    # Convolutional layer
    conv = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)

    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    flat = Flatten()(pool)
    # always use last nb_neurons value for dense layer

    # Fully connected layers
    dense = Dense(128, activation='relu')(flat)

    # Output layers for task 1
    task1_output = Dense(nb_classes, activation='softmax', name='task1_output')(dense)

    # Output layers for task 2
    task2_output = Dense(nb_classes, activation='softmax', name='task2_output')(dense)

    # Create the model
    model = Model(inputs=inputs, outputs=[task1_output, task2_output])

    # Compile the model
    model.compile(optimizer='adam', loss={'task1_output': 'categorical_crossentropy',
                                          'task2_output': 'categorical_crossentropy'},
                  metrics={'task1_output': 'accuracy', 'task2_output': 'accuracy'})
    # model.fit(x_train, [y_train1, y_train2], batch_size=32, epochs=10, validation_data=(x_test, [y_test1, y_test2]))

    visualkeras.layered_view(model, legend=True, to_file=f'test_multi.png')


if __name__ == '__main__':
    # plot_sequential_model();
    plot_sequential_model();
