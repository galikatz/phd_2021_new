from sklearn.utils import shuffle
import numpy as np


def create_balanced_incong_cong_train_test(x_incong_stimuli, y_incong_labels, x_cong_stimuli, y_cong_labels):
	# wrap in touples before shuffeling to incong and cong:
	list_of_touples = []
	for i in range(0,len(x_incong_stimuli)):
		touple_of_stiuli_and_label = (x_incong_stimuli[i], y_incong_labels[i])
		list_of_touples.append(touple_of_stiuli_and_label)

	for i in range(0, len(x_cong_stimuli)):
		touple_of_stiuli_and_label = (x_cong_stimuli[i], y_cong_labels[i])
		list_of_touples.append(touple_of_stiuli_and_label)

	print('before shuffle {}'.format(list_of_touples))
	shuffled = shuffle(list_of_touples)
	print('after shuffle {}'.format(shuffled))

	# unwrap the toupling
	x = []
	y = []

	for i in range(0, len(shuffled)):
		stimuli = shuffled[i][0]
		label = shuffled[i][1]
		x.append(stimuli)
		y.append(label)

	return np.array(x), np.array(y)


if __name__ == '__main__':

	x_incong_train = [2,4,6,8]
	y_incong_train = [22, 44, 66, 88]
	x_cong_train = [1,3,5,7]
	y_cong_train = [11,33,55,77]

	x_train, y_train = create_balanced_incong_cong_train_test(x_incong_train, y_incong_train, x_cong_train, y_cong_train)
	print ('x_train: {}'.format(x_train))
	print('y_train: {}'.format(y_train))


