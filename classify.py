import argparse
from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import glob
import logging
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D

IMG_SIZE = 100

def main(args):
	(x_train, y_train), (x_test, y_test) = creating_train_test_data(args.images_dir, args.stimuli_type, args.mode)
	create_model(x_train, y_train, x_test, y_test)


def classify_labels_according_to_mode(mode, path, epochs):
	nb_classes = 2
	batch_size = 60

	# Input image dimensions
	input_shape = (IMG_SIZE, IMG_SIZE, 3)#RGB
	(x_train, y_train), (x_test, y_test) = creating_train_test_data(path, "katzin", mode)
	y_train = to_categorical(y_train, nb_classes)
	y_test = to_categorical(y_test, nb_classes)
	print('y_train size: {} y_test size: {}'.format(len(y_train), len(y_test)))
	return nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs


def create_model(x_train, y_train, x_test, y_test):
	# one hot encoding outputs
	y_train = np_utils.to_categorical(y_train)#converting to binary
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


def extract_label(file_name, stimuli_type, task):
	if stimuli_type == 'halberda':
		description = file_name[file_name.rindex('/')+1:file_name.index('.jpg')]
		labels = description.split('_')
		return {'classification_label': labels[0], 'yellow_num': labels[1], 'blue_num': labels[2]}
	else:
		description = file_name[file_name.rindex('/') + 1:file_name.index('.jpg')]

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
					classification_label = '1'#right is bigger in count
				else:
					classification_label = '0'
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
					classification_label = '1'
				else:
					classification_label = '0'
		if task == 'random':
			random_result = random.random()
			if random_result > 0.5:
				classification_label = '0' # left stimulus is bigger
			else:
				classification_label = '1'  # right stimulus is bigger

		return {'congruency': congruency, 'ratio': ratio, 'left_num': labels[1], 'right_num': labels[2], 'classification_label': classification_label}


#########################################
# Labeling and creating / switching tasks
#########################################
def creating_train_test_data(dir, stimuli_type, mode):
	logging.info("*** Classifying and labeling accroding to mode %s ***" % mode)
	data = []
	files = glob.glob(dir + os.sep + '*.jpg')
	logging.info("got %s files to train " % len(files))
	# this is for the first time basically
	task = mode
	for path in files:
		#print(path)
		# Classification: returns 0 if left stimuli is more white/ bigger numerically and 1 otherwise.
		label = extract_label(path, stimuli_type, task)
		rgba_image = Image.open(path)
		rgb_image = rgba_image.convert('RGB')
		rgb_image = rgb_image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
		img = rgb_image.copy()
		data.append([np.array(img), label['classification_label']])

	# shuffle the collection
	data = shuffle(data)
	final_data = []
	final_labels = []
	for entry in data:
		image = entry[0]
		label = entry[1]
		final_data.append(image)
		final_labels.append(label)
	final_data_after_normalization = np.array(final_data, dtype="float") / 255.0
	final_labels_as_np_array = np.array(final_labels)
	(x_train, x_test, y_train, y_test) = train_test_split(final_data_after_normalization, final_labels_as_np_array, test_size=0.2, random_state=101)
	return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='classifier arguments')
	parser.add_argument('--images_dir', dest='images_dir', type=str, required=True, help='The images dir')
	parser.add_argument('--stimuli_type', dest='stimuli_type', type=str, required=True, help='The stimuli type')
	parser.add_argument('--mode', dest='mode', type=str, required=True, help='task mode (size/count/both)')
	args = parser.parse_args()
	main(args)