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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
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
from classify import classify_labels_according_to_mode
from datetime import datetime
from keras.utils.vis_utils import plot_model
from evolution_utils import DataPerSubject
matplotlib.use('TkAgg')


# Helper: Early stopping.
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto')

class TrainClassificationCache:
	def __init__(self):
		self.nb_classes = None
		self.batch_size = None
		self.input_shape = None
		self.x_train = None
		self.x_test = None
		self.x_test = None
		self.y_train = None
		self.y_test = None
		self.x_cong_test = None
		self.y_cong_test = None
		self.x_incong_test = None
		self.y_incong_test = None
		self.cache_is_empty = True


	def update_classification_cache(self, nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, x_cong_test, y_cong_test, x_incong_test, y_incong_test):
		self.nb_classes = nb_classes
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.x_train = x_train
		self.x_test = x_test
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test
		self.x_cong_test = x_cong_test
		self.y_cong_test = y_cong_test
		self.x_incong_test = x_incong_test
		self.y_incong_test = y_incong_test
		self.cache_is_empty = False


def refresh_classification_cache():
	return TrainClassificationCache()

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

	bce = losses.BinaryCrossentropy(reduction='none')
	model.compile(loss=bce,
				  optimizer=optimizer,
				  metrics=['accuracy'])

    # calculate the loss of congruent and incongruent sepatately
 	# bce = tf.keras.losses.BinaryCrossentropy(reduction='none')
	# bce(y_true, y_pred).numpy()
	# array([0.9162905, 0.5919184, 0.79465103, 1.0549198], dtype=float32)

	# model.compile(loss='categorical_crossentropy',
	# 			  optimizer=optimizer,
	# 			  metrics=['accuracy'])

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


def train_and_score(genome, dataset, mode, path, batch_size, epochs, debug_mode, max_val_accuracy, trainer_classification_cache, model=None):

	logging.info("Preparing stimuli")
	if dataset == 'size_count':
		if not trainer_classification_cache.cache_is_empty:
			nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, x_cong_test, y_cong_test, x_incong_test, y_incong_test = trainer_classification_cache.nb_classes, \
																					trainer_classification_cache.batch_size,\
																					trainer_classification_cache.input_shape, \
																					trainer_classification_cache.x_train, \
																					trainer_classification_cache.x_test, \
																					trainer_classification_cache.y_train, \
																					trainer_classification_cache.y_test, \
																					trainer_classification_cache.x_cong_test, \
																					trainer_classification_cache.y_cong_test, \
																					trainer_classification_cache.x_incong_test, \
																					trainer_classification_cache.y_incong_test
		else:
			nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, x_cong_test, y_cong_test, x_incong_test, y_incong_test = classify_labels_according_to_mode(mode, path, batch_size)
			trainer_classification_cache.update_classification_cache(nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, x_cong_test, y_cong_test, x_incong_test, y_incong_test)


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

	score = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=0)

	# taking the last epocj result to be kept ( and not all the loss and accuracies from all epochs, since the last epoch is the best)
	training_accuracy = history.history["accuracy"][-1]
	validation_accuracy = history.history["val_accuracy"][-1]
	training_loss = history.history["loss"][-1]
	validation_loss = history.history["val_loss"][-1]

	score_congruent = model.evaluate(x=x_cong_test, y=y_cong_test, batch_size=batch_size, verbose=0)
	score_incongruent = model.evaluate(x=x_incong_test, y=y_incong_test, batch_size=batch_size, verbose=0)

	validation_accuracy_congruent = score_congruent[1]
	validation_accuracy_incongruent = score_incongruent[1]
	validation_loss_congruent = score_congruent[0]
	validation_loss_incongruent = score_incongruent[0]

	data_per_subject = DataPerSubject(genome.u_ID,
				 training_accuracy,
				 validation_accuracy,
				 training_loss,
				 validation_loss,
				 validation_accuracy_congruent,
				 validation_accuracy_incongruent,
				 validation_loss_congruent,
				 validation_loss_incongruent)

	training_set_size = len(x_train)
	validation_set_size = len(x_test)
	validation_set_size_congruent = len(x_cong_test)

	#savig the results of each prediction
	y_test_prediction = model.predict(x=x_test, batch_size=batch_size, verbose=0)

	#fixing the prediction result to be 0 and 1 and not float thresholds.
	y_test_corrected = []
	for i in range(len(y_test_prediction)):
		if y_test_prediction[i][0] > 0.5:
			left_stimulus_result = 1
			right_stimulus_result = 0
		else:
			left_stimulus_result = 0
			right_stimulus_result = 1
		y_test_corrected.append(np.array([left_stimulus_result,right_stimulus_result]))

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
	return best_current_val_accuracy, best_current_val_loss, y_test_corrected, model, data_per_subject, training_set_size, validation_set_size, validation_set_size_congruent


class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
