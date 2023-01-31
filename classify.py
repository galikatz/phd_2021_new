import argparse
from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import glob
import logging
import random
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from evolution_utils import RATIOS
from train_test_data import TrainTestData

IMG_SIZE = 100


def create_model(x_train, y_train, x_test, y_test):
	# one hot encoding outputs
	y_train = np_utils.to_categorical(y_train)  #converting to binary
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]

	# Create the model
	model = Sequential()

	model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Flatten())
	model.add(Dropout(0.2))

	model.add(Dense(256, kernel_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(128, kernel_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	epochs = 25
	optimizer = 'Adam'

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	print(model.summary())

	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64)

	# Final evaluation of the model

	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1] * 100))


def extract_label(file_name, stimuli_type, task, one_hot):
	if stimuli_type == 'halberda':
		description = file_name[file_name.rindex('/')+1:file_name.index('.jpg')]
		labels = description.split('_')
		return {'classification_label': labels[0], 'yellow_num': labels[1], 'blue_num': labels[2]}
	else:
		description = file_name[file_name.rindex(os.sep) + 1:file_name.index('.jpg')]

		if 'incong' in description:
			labels = description.split('_')
			ratio = labels[0].replace('incong','')
			congruency = 0
			left_num = int(labels[1])
			right_num = int(labels[2])
			if task == 'size':
				if left_num < right_num: #number is lower but the area is bigger since this is incongruent (left=0, right=1)
					classification_label = '0'#left is bigger in size
				else:
					classification_label = '1'
			elif task == 'count':
				if left_num < right_num:#(left=0, right=1)
					classification_label = right_num if one_hot else 1#'1'#right is bigger in count
				else:
					classification_label = left_num if one_hot else 0#'0'
			elif task == 'colors':  # Where is Cyan?
				left_color = labels[7]
				if left_color == 'c':
					classification_label = '0'  # left stimulus is cyan
				else:
					classification_label = '1'  # right stimulus is cyan
		else:#congruent
			labels = description.split('_')
			ratio = labels[0].replace('cong', '')
			congruency = 1
			left_num = int(labels[1])
			right_num = int(labels[2])
			if task == 'size':
				if left_num < right_num:  # number is lower and the area is lower since this is congruent (left=0, right=1)
					classification_label = '1'
				else:
					classification_label = '0'
			elif task == 'count':
				if left_num < right_num:  # (left=0, right=1)
					classification_label = right_num if one_hot else 1#'1'
				else:
					classification_label = left_num if one_hot else 0#'0'
			elif task == 'colors':  # Where is Cyan?
				left_color = labels[7]
				if left_color == 'c':
					classification_label = '0'  # left stimulus is cyan
				else:
					classification_label = '1'  # right stimulus is cyan

		return {'congruency': congruency, 'ratio': ratio, 'left_num': labels[1], 'right_num': labels[2], 'classification_label': classification_label}


#########################################
# Labeling and creating / switching tasks
#########################################
def creating_train_test_data(dir, stimuli_type, mode, nb_classes, one_hot):
	logging.info("########## Classifying and labeling accroding to mode %s #########" % mode)

	incong_files = glob.glob(dir + os.sep + 'incong*.jpg')
	cong_files = glob.glob(dir + os.sep + 'cong*.jpg')
	logging.info("Got %s incong and %s cong files to train from path %s" % (len(incong_files), len(cong_files), dir))
	# this is for the first time basically
	(x_incong_train, x_incong_test, y_incong_train, y_incong_test) = classify_and_split_to_train_test(mode, incong_files, stimuli_type, one_hot)
	(x_cong_train, x_cong_test, y_cong_train, y_cong_test) = classify_and_split_to_train_test(mode, cong_files, stimuli_type, one_hot)

	# main dataset
	x_train, y_train = create_balanced_incong_cong_train_test(x_incong_train, y_incong_train, x_cong_train, y_cong_train)
	x_test, y_test = create_balanced_incong_cong_train_test(x_incong_test, y_incong_test, x_cong_test, y_cong_test)

	# congruency dataset
	x_cong_train, y_cong_train = create_balanced_incong_cong_train_test(x_cong_train, y_cong_train, [], [])
	x_incong_train, y_incong_train= create_balanced_incong_cong_train_test([], [], x_incong_train, y_incong_train)
	x_cong_test, y_cong_test = create_balanced_incong_cong_train_test(x_cong_test, y_cong_test, [], [])
	x_incong_test, y_incong_test = create_balanced_incong_cong_train_test([], [], x_incong_test, y_incong_test)

	y_train = to_categorical(y_train, nb_classes)
	y_test = to_categorical(y_test, nb_classes)
	y_cong_train = to_categorical(y_cong_train, nb_classes)
	y_incong_train = to_categorical(y_incong_train, nb_classes)
	y_cong_test = to_categorical(y_cong_test, nb_classes)
	y_incong_test = to_categorical(y_incong_test, nb_classes)

	ratios_training_dataset = {}
	ratios_validation_dataset = {}
	for ratio in RATIOS:
		ratio_cong_files = glob.glob(dir + os.sep + 'cong' + str(ratio) + '*.jpg')
		ratio_incong_files = glob.glob(dir + os.sep + 'incong' + str(ratio) + '*.jpg')
		logging.info("Got ratio: %s cong files: %s to train and incong files: %s to train" % (str(ratio), len(ratio_cong_files), len(ratio_incong_files)))
		(x_ratio_cong_train, x_ratio_cong_test, y_ratio_cong_train, y_ratio_cong_test) = classify_and_split_to_train_test(mode, ratio_cong_files, stimuli_type, one_hot)
		(x_ratio_incong_train, x_ratio_incong_test, y_ratio_incong_train, y_ratio_incong_test) = classify_and_split_to_train_test(mode, ratio_incong_files, stimuli_type, one_hot)


		x_ratio_cong_train, y_ratio_cong_train = create_balanced_incong_cong_train_test(x_ratio_cong_train, y_ratio_cong_train, [], [])
		x_ratio_incong_train, y_ratio_incong_train = create_balanced_incong_cong_train_test([], [], x_ratio_incong_train, y_ratio_incong_train)

		x_ratio_cong_test, y_ratio_cong_test = create_balanced_incong_cong_train_test(x_ratio_cong_test, y_ratio_cong_test, [], [])
		x_ratio_incong_test, y_ratio_incong_test = create_balanced_incong_cong_train_test([], [], x_ratio_incong_test, y_ratio_incong_test)

		# Fix Y-label values
		y_ratio_cong_train = to_categorical(y_ratio_cong_train, nb_classes)
		y_ratio_incong_train = to_categorical(y_ratio_incong_train, nb_classes)

		y_ratio_cong_test = to_categorical(y_ratio_cong_test, nb_classes)
		y_ratio_incong_test = to_categorical(y_ratio_incong_test, nb_classes)

		ratios_training_dataset.update({ratio: [(x_ratio_cong_train, y_ratio_cong_train), (x_ratio_incong_train, y_ratio_incong_train)]})

		ratios_validation_dataset.update({ratio: [(x_ratio_cong_test, y_ratio_cong_test), (x_ratio_incong_test, y_ratio_incong_test)]})

	train_test_data = TrainTestData(ratios_training_dataset, ratios_validation_dataset,
                 x_train, y_train,
                 x_test, y_test,
                 x_cong_train, y_cong_train,
                 x_incong_train, y_incong_train,
                 x_cong_test, y_cong_test,
                 x_incong_test, y_incong_test)
	return train_test_data


def create_balanced_incong_cong_train_test(x_incong_stimuli, y_incong_labels, x_cong_stimuli, y_cong_labels):
	# wrap in touples before shuffeling to incong and cong:
	list_of_touples = []
	for i in range(0,len(x_incong_stimuli)):
		touple_of_stimuli_and_label = (x_incong_stimuli[i], y_incong_labels[i])
		list_of_touples.append(touple_of_stimuli_and_label)

	for i in range(0, len(x_cong_stimuli)):
		touple_of_stiuli_and_label = (x_cong_stimuli[i], y_cong_labels[i])
		list_of_touples.append(touple_of_stiuli_and_label)

	shuffled = shuffle(list_of_touples)
	# unwrap the toupling
	x = []
	y = []

	for i in range(0, len(shuffled)):
		stimuli = shuffled[i][0]
		label = shuffled[i][1]
		x.append(stimuli)
		y.append(label)

	return np.array(x), np.array(y)


def classify_and_split_to_train_test(mode, files, stimuli_type, one_hot):
	data = []
	task = mode
	for path in files:
		# print(path)
		# Classification: returns 0 if left stimuli is more white/ bigger numerically and 1 otherwise.
		label = extract_label(path, stimuli_type, task, one_hot)
		rgba_image = Image.open(path)
		rgb_image = rgba_image.convert('RGB')
		rgb_image = rgb_image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
		img = rgb_image.copy()
		data.append([np.array(img), label['classification_label']])

	final_data = []
	final_labels = []
	for entry in data:
		image = entry[0]
		label = entry[1]
		final_data.append(image)
		final_labels.append(label)
	final_data_after_normalization = np.array(final_data, dtype="float") / 255.0
	final_labels_as_np_array = np.array(final_labels)
	(x_train, x_test, y_train, y_test) = train_test_split(final_data_after_normalization, final_labels_as_np_array, test_size=0.2, random_state=101, shuffle=True)
	return x_train, x_test, y_train, y_test
