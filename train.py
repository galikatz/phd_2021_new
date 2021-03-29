"""
Generic setup of the data sources and the model training. 

Based on:
	https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
	https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""

import random
import glob
import cv2
from imutils import paths
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from classify import get_size_count_new
from datetime import datetime
from keras.utils.vis_utils import plot_model
matplotlib.use('TkAgg')


# Helper: Early stopping.
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto')


def load_size_count_data(mode, path, current_generation):
	"""Loads the Size Count dataset.

	# Arguments
	  the path to the dataset.

	# Returns
		Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
	"""
	imagePaths = sorted(list(paths.list_images(path)))
	random.seed(42)
	random.shuffle(imagePaths)
	data = []
	counting_labels = []
	size_labels = []
	# loop over the input images
	for imagePath in imagePaths:
		# load the image, pre-process it, and store it in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (100, 100))
		image = img_to_array(image)
		data.append(image)

		# extract the class label from the image path and update the
		# labels list
		image_name = imagePath.split(os.path.sep)[-1]
		image_name = image_name[0:image_name.rindex('.')]
		file_name_arr = image_name.split('_')
		size_perception_label = file_name_arr[0]
		size_labels.append(size_perception_label)
		counting_label = file_name_arr[1]
		counting_labels.append(counting_label)

	# scale the raw pixel intensities to the range [0, 1]
	final_data = np.array(data, dtype="float") / 255.0
	if mode == 'size':
		final_labels = np.array(size_labels)
	else:
		final_labels = np.array(counting_labels)

	(x_train, x_test, Y_train, Y_test) = train_test_split(final_data, final_labels, test_size=0.2, random_state=101)
	return (x_train, Y_train), (x_test, Y_test)


def prepare_and_plot_dist(mode_name, Y_train, Y_test, current_generation):
	df_train = pd.DataFrame()
	df_train['train'] = Y_train
	df_train['label'] = df_train['train'].apply(convert_to_labels)

	df_test = pd.DataFrame()
	df_test['test'] = Y_test

	train_df_counts = df_train['train'].value_counts()
	test_df_counts = df_test['test'].value_counts()

	if mode_name == 'size' or mode_name == 'both':
		data = [['few', 'train', train_df_counts[0]],
				['few', 'test', test_df_counts[0]],
				['many', 'train', train_df_counts[1]],
				['many', 'test', test_df_counts[1]]]
		viz_df = pd.DataFrame(data, columns=['label', 'mode', 'count'])
		plot_dist(viz_df, 'size or both', 'Stimuli distribution Gen' + str(current_generation))

	data = []
	if mode_name == 'count' or mode_name == 'both':
		for i in range(0, len(train_df_counts)):
			data.append([str(i), 'train', train_df_counts[i]])
		for j in range(0, len(test_df_counts)):
			data.append([str(j), 'test', test_df_counts[j]])
		viz_df = pd.DataFrame(data, columns=['label', 'mode', 'count'])
		plot_dist(viz_df, 'count or both', 'Stimuli distribution Gen' + str(current_generation))


def convert_to_labels(x):
	if x == '0':
		return 'few'
	else:
		return 'many'


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


def get_size_count(mode, path, epochs, current_generation):
	# Set defaults.
	nb_classes = None
	if mode == 'size':
		nb_classes = 2
	else:
		nb_classes = 11
	batch_size = 128

	# Input image dimensions
	img_rows, img_cols = 100, 100

	# Get the data.
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = load_size_count_data(mode, path, current_generation)

	# if K.image_data_format() == 'channels_first':
	#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	#     input_shape = (1, img_rows, img_cols)
	# else:
	#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	#     input_shape = (img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 3)  # RGB
	# x_train = x_train.reshape(60000, 784)
	# x_test  = x_test.reshape(10000, 784)

	# x_train = X_train.astype('float32')
	# x_test = X_test.astype('float32')
	# x_train /= 255
	# x_test /= 255

	# print('x_train shape:', x_train.shape)
	# print(x_train.shape[0], 'train samples')
	# print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = to_categorical(y_train, nb_classes)
	y_test = to_categorical(y_test, nb_classes)

	# convert class vectors to binary class matrices
	# y_train = keras.utils.to_categorical(y_train, nb_classes)
	# y_test = keras.utils.to_categorical(y_test, nb_classes)

	return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test, epochs)


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

	# Add each layer.
	for i in range(0, len(nb_neurons)):
		# Need input shape for first layer.
		if i == 0:
			model.add(Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation, padding='same',
							 input_shape=input_shape))
		else:
			print("index is {} and num_neurons is {} and nb_layers size is {}".format(i, len(nb_neurons), nb_layers))
			model.add(Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation))

		if i < 2:  # otherwise we hit zero
			model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Dropout(0.2))

	model.add(Flatten())
	# always use last nb_neurons value for dense layer
	model.add(Dense(nb_neurons[len(nb_neurons) - 1], activation=activation))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))

	# BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE

	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	return model


def plot_genome_after_training_on_epochs_is_done(genome, mode, epochs, val_acc, val_loss, train_acc, train_loss, date, best_accuracy, best_loss):
	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	# using in case it stopped with early stopping
	plt.plot(np.arange(1,epochs+1), train_loss, label="train_loss")
	plt.plot(np.arange(1,epochs+1), val_loss, label="val_loss")
	plt.plot(np.arange(1,epochs+1), train_acc, label="train_acc")
	plt.plot(np.arange(1,epochs+1), val_acc, label="val_acc")
	plt.title("Training Loss and Accuracy: {mode}".format(mode=mode))
	plt.xlabel("Epochs")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="upper left")
	plt.savefig("models/best_model_{}_mode_{}_gen_{}_individual_{}_acc_{}_loss_{}.jpg".format(date, mode, genome.generation, genome.u_ID, best_accuracy, best_loss))


def train_and_score(genome, dataset, mode, path, epochs, debug_mode, mode_th, max_val_accuracy, min_val_loss, model=None):
	"""Train the model, return test loss.
	Args:
		network (dict): the parameters of the network
		dataset (str): Dataset to use for training/evaluating

	"""
	logging.info("Preparing stimuli")

	if dataset == 'size_count':
		nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_size_count_new(mode, path, epochs, mode_th, max_val_accuracy)

	if not model:
		logging.info("*********** Creating a new Keras model for individual %s ***********" % genome.u_ID)

		if dataset == 'size_count':
			model = compile_model_cnn(genome, nb_classes, input_shape)
	else:
		logging.info("*********** Using the existing model for individual %s ***********" % genome.u_ID)

	# prints the model
	model.summary()

	history = LossHistory()

	history = model.fit(x_train, y_train,
						batch_size=batch_size,
						epochs=epochs,
						verbose=1,
						validation_data=(x_test, y_test),
						callbacks=[history,early_stopper]) # using early stopping so no real limit - don't want to waste time on horrible architectures

	score = model.evaluate(x_test, y_test, verbose=0)

	best_current_val_loss = round(score[0],3)
	best_current_val_accuracy = round(score[1],3)
	print('Best current test loss from all epochs:', best_current_val_loss) # takes the minimum loss from all the epochs
	print('Best current test accuracy from all epochs based on minimal loss:', best_current_val_accuracy) # taking the accuracy of the minimal loss above.

	############################################################################################################################
	# Saving the best model from all individuals and deleting the unecessary ones, but will return the current individual result.
	############################################################################################################################
	# we save the model only if the accuracy is better than what we currently have
	if max_val_accuracy < best_current_val_accuracy:
		max_val_accuracy = best_current_val_accuracy
		min_val_loss = best_current_val_loss
		# serialize model to JSON
		model_json = model.to_json()
		date = datetime.strftime(datetime.now(), '%Y-%m-%d_%H')
		if mode == 'both':
			mode_name = 'size'
		else:
			mode_name = mode

		#delete old files from today because we have better results:
		old_models_path = "models/best_model_{}_mode_{}*.*".format(date, mode, genome.generation, genome.u_ID)
		for f in glob.glob(old_models_path):
			os.remove(f)

		with open("models/best_model_{}_mode_{}_gen_{}_individual_{}_acc_{}_loss_{}.json".format(date, mode_name, genome.generation, genome.u_ID, max_val_accuracy, min_val_loss),"w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		file_name = "models/best_model_{}_mode_{}_gen_{}_individual_{}_acc_{}_loss_{}".format(date, mode_name, genome.generation, genome.u_ID, max_val_accuracy, min_val_loss)
		model.save(file_name + ".h5")
		if debug_mode:
			# this is the list of all the epochs scores of the current generation - for plotting
			train_loss = history.history["loss"]
			val_loss = history.history["val_loss"]
			train_acc = history.history["accuracy"]
			val_acc = history.history["val_accuracy"]
			#plot_genome_after_training_on_epochs_is_done(genome, mode_name, epochs, val_acc, val_loss, train_acc, train_loss, date, max_val_accuracy, min_val_loss)
			#plot_model(model, to_file=file_name+'.png', show_shapes=True, show_layer_names=True)


	K.clear_session()
	# getting only the last values
	return best_current_val_accuracy, best_current_val_loss, model


class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
