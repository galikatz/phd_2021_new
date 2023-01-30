import matplotlib.pyplot as plt

import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
from keras import backend as K
import numpy as np
import os
import logging
import seaborn as sns
from datetime import datetime
from classify import creating_train_test_data, IMG_SIZE
from evolution_utils import evaluate_model

# Helper: Early stopping.
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto')
FIXED_NB_CLASSES = 2


class TrainClassificationCache:
    def __init__(self):
        self.nb_classes = None
        self.batch_size = None
        self.input_shape = None
        self.train_test_data = None
        self.cache_is_empty = True

    def update_classification_cache(self, nb_classes, batch_size, input_shape, train_test_data):
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.train_test_data = train_test_data
        self.cache_is_empty = False


def refresh_classification_cache():
    return TrainClassificationCache()


def plot_dist(df, mode_name, title):
    df.info()
    df.describe()
    plt.figure(figsize=(8, 8))
    sns_t = sns.barplot(x='label', y='count', hue='mode', data=df, palette='rainbow', ci=None)
    # show_values_on_bars(sns_t, "v", -15)
    plt.title(title)
    plt.savefig(title + ' ' + mode_name + '.png')
    plt.close()


def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + float(space)
                val = p.get_height()
                value = int(val)
                ax.text(_x, _y, value, ha="center", size=4)

        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left", size=4)

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def compile_multitask_cnn(genome, nb_classes, input_shape):
    logging.info("********* Creating a new multitask CNN model **********")
    nb_layers = genome.geneparam['nb_layers']
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer = genome.geneparam['optimizer']

    logging.info("Architecture:%s,%s,%s,%d" % (str(nb_neurons), activation, optimizer, nb_layers))
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
    from keras.models import Model

    # Convolutional layers
    for i in range(0, len(nb_neurons)):
        # Need input shape for first layer.
        if i == 0:
            x = Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation, padding='same',
                             input_shape=input_shape)
        else:
            x = Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation)(x)

        if i < 2:  # otherwise we hit zero
            x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.2)(x)

    x = Flatten()(x)
    # always use last nb_neurons value for dense layer
    x = Dense(nb_neurons[len(nb_neurons) - 1], activation=activation)(x)
    x = Dropout(0.5)(x)
    # Fully connected layers
    x = Dense(128, activation='relu')(x)

    # Output layers for task 1
    task1_output = Dense(nb_classes, activation='softmax', name='task1_output')(x)

    # Output layers for task 2
    task2_output = Dense(nb_classes, activation='softmax', name='task2_output')(x)

    # Create the model
    model = Model(inputs=input_shape, outputs=[task1_output, task2_output])

    # Compile the model
    model.compile(optimizer='adam', loss={'task1_output': 'categorical_crossentropy',
                                         'task2_output': 'categorical_crossentropy'},
                  metrics={'task1_output': 'accuracy', 'task2_output': 'accuracy'})
    return model


def compile_model_cnn(genome, nb_classes, input_shape):
    """Compile a sequential model.

	Args:
		genome (dict): the parameters of the genome

	Returns:
		a compiled network.

	"""
    # Get our network parameters.
    logging.info("********* Creating a new CNN model **********")

    nb_layers = genome.geneparam['nb_layers']
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer = genome.geneparam['optimizer']

    logging.info("Architecture:%s,%s,%s,%d" % (str(nb_neurons), activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer. this len(nb_neurons)) is a simplification, saying we will have only 3 layers (const) and there will not be an evolution on the number of layers.
    for i in range(0, nb_layers): #len(nb_neurons)
        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation, padding='same',
                             input_shape=input_shape))
        else:
            model.add(Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation))

        if i < 2:  # otherwise we hit zero
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

    model.add(Flatten())

    # Fully connected layers
    # always use last nb_neurons value for dense layer
    model.add(Dense(nb_neurons[len(nb_neurons) - 1], activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    # BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE

    bce = losses.BinaryCrossentropy(reduction='none')
    model.compile(loss=bce,
                  optimizer=optimizer,
                  metrics=["accuracy"])

    return model


def plot_genome_after_training_on_epochs_is_done(genome, mode, epochs, val_acc, val_loss, train_acc, train_loss, date,
                                                 best_accuracy, best_loss):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    # using in case it stopped with early stopping
    plt.plot(np.arange(1, epochs + 1), train_loss, label="train_loss")
    plt.plot(np.arange(1, epochs + 1), val_loss, label="val_loss")
    plt.plot(np.arange(1, epochs + 1), train_acc, label="train_acc")
    plt.plot(np.arange(1, epochs + 1), val_acc, label="val_acc")
    plt.title("Training Loss and Accuracy: {mode}".format(mode=mode))
    plt.xlabel("Epochs")
    plt.ylabel("Loss " + os.sep + "Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(
        "models" + os.sep + "best_model_{}_mode_{}_gen_{}_individual_{}_acc_{}_loss_{}.jpg".format(date, mode, genome.generation,
                                                                                      genome.u_ID, best_accuracy,
                                                                                      best_loss))


def train_and_score(genome, dataset, mode, equate, path, batch_size, epochs, debug_mode, max_val_accuracy,
                    trainer_classification_cache, model=None, training_strategy=None):
    logging.info("Preparing stimuli")
    input_shape = (IMG_SIZE, IMG_SIZE, 3)#RGB
    nb_classes = FIXED_NB_CLASSES
    if dataset == 'size_count':
        if not trainer_classification_cache.cache_is_empty:
            train_test_data = trainer_classification_cache.train_test_data
        else:
            train_test_data = creating_train_test_data(
                dir=path, stimuli_type="katzin", mode=mode, nb_classes=nb_classes)
            trainer_classification_cache.update_classification_cache(nb_classes, batch_size, input_shape, train_test_data)

    if not model:
        logging.info("*********** Creating a new Keras model for individual %s ***********" % genome.u_ID)

        if dataset == 'size_count':
            if training_strategy is not None:
                with training_strategy.scope():
                    model = compile_model_cnn(genome, nb_classes, input_shape)
            else:
                model = compile_model_cnn(genome, nb_classes, input_shape)
    else:
        logging.info("*********** Using the existing model for individual %s ***********" % genome.u_ID)

    # prints the model
    model.summary()

    history = LossHistory()

    history = model.fit(train_test_data.x_train, train_test_data.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(train_test_data.x_test, train_test_data.y_test),
                        callbacks=[history,
                                   early_stopper])  # using early stopping so no real limit - don't want to waste time on horrible architectures
    train_result = evaluate_model(genome=genome, model=model, history=history, train_test_data=train_test_data, batch_size=batch_size)

    ############################################################################################################################
    # Saving the model from all individuals and deleting the unnecessary ones, but will return the current individual result.
    ############################################################################################################################

    date = datetime.strftime(datetime.now(), '%Y-%m-%d_%H')
    if mode == 'both':
        mode_name = 'size'
    else:
        mode_name = mode

    # delete old files of past generations leaving only the current one.
    old_models_path = "models" + os.sep + "*mode_{}_equate_{}_gen_{}*.*".format(mode, equate, genome.generation-1)
    for f in glob.glob(old_models_path):
        os.remove(f)

    filename = ("models" + os.sep + "model_{}_mode_{}_equate_{}_gen_{}_individual_{}_acc_{}_loss_{}_layers_{}_neurons_{}_activation_{}_optimizer_{}").format(date, mode_name,
                                                                                 equate,
                                                                                 genome.generation,
                                                                                 genome.u_ID,
                                                                                 train_result.curr_individual_acc,
                                                                                 train_result.curr_individual_loss,
                                                                                 genome.geneparam['nb_layers'],
                                                                                 genome.nb_neurons(),
                                                                                 genome.geneparam['activation'],
                                                                                 genome.geneparam['optimizer'])
    ####################################################################################################################################
    # Save all models not the best ones.
    # 1. the model's configuration (topology)
    # 2. the model's weights
    # 3. the model's optimizer's state (if any)
    # Thus models can be reinstantiated in the exact same state, without any of the code used for model definition or training.
    ####################################################################################################################################
    model.save(filename + ".h5")
    K.clear_session()
    return train_result


def append_to_main_test_result(main_train_result, other_stimuli_train_result):
    pass


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
