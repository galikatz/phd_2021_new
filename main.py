"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function

from evolver import Evolver
import idgen
from stimuli_generator import create_stimuli

from tqdm import tqdm

import logging
import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt


# Setup logging.
logging.basicConfig(
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%m/%d/%Y %I:%M:%S %p',
	level=logging.INFO#,
	#filename='log.txt'
)


def train_genomes(genomes, dataset, mode, path, epochs, debug_mode, mode_th):
	"""Train each genome.

	Args:
		networks (list): Current population of genomes
		dataset (str): Dataset to use for training/evaluating

	"""
	logging.info("*** train_networks(networks, dataset) ***")
	logging.info("*** total population size: *** %s" % len(genomes))
	pbar = tqdm(total=len(genomes))
	for genome in genomes:
		genome.train(dataset, mode, path, epochs, debug_mode, mode_th)
		pbar.update(1)

	pbar.close()

def get_average_accuracy(genomes):
	"""Get the average accuracy for a group of networks/genomes.

	Args:
		networks (list): List of networks/genomes

	Returns:
		float: The average accuracy of a population of networks/genomes.

	"""
	total_accuracy = 0

	for genome in genomes:
		total_accuracy += genome.accuracy

	return total_accuracy / len(genomes)


def get_best_genome(genomes):
	"""
	Gets the best individual in this generation
	:param genomes:
	:return: the accuracy score
	"""
	genomes_dict={}
	max_accuracy = 0.0
	for genome in genomes:
		genomes_dict.update({genome.accuracy: genome})
		max_accuracy = max(max_accuracy, genome.accuracy)

	best_genome = genomes_dict.get(max_accuracy)
	logging.info("best genome has %f accuracy "%(best_genome.accuracy))
	return best_genome


def generate(generations, population, all_possible_genes, dataset, mode, mode_th, images_dir, stopping_th, epochs, debug_mode, genomes=None, evolver=None):
	"""Generate a network with the genetic algorithm.

	Args:
		generations (int): Number of times to evolve the population
		population (int): Number of networks in each generation
		all_possible_genes (dict): Parameter choices for networks
		dataset (str): Dataset to use for training/evaluating

	"""
	logging.info("*** mode={}, mode_th={}, generations={}, population={}, epochs={}, stopping_th={})***".format(mode, mode_th, generations, population, epochs, stopping_th))

	if not genomes:
		evolver = Evolver(all_possible_genes)
		genomes = evolver.create_population(population)

	# Evolve the generation.
	already_switched = False
	for i in range(1, generations):
		images_dir_per_gen = images_dir
		logging.info("*** Now in generation %d of %d reading images from dir: %s ***" % (i, generations, images_dir_per_gen))

		print_genomes(genomes)
		train_genomes(genomes, dataset, mode, images_dir_per_gen, epochs, debug_mode, mode_th)

		# Get the average accuracy for this generation.
		average_accuracy = get_average_accuracy(genomes)

		if mode != "both" and average_accuracy >= stopping_th:
			logging.info("Done training! average_accuracy is %s" % str(average_accuracy))
			break

		if mode == "both": # this is for the first time before the switch (no recursion)
			if average_accuracy >= mode_th:
				if average_accuracy >= stopping_th:
						if already_switched:
							logging.info("Done training! average_accuracy is %s" % str(average_accuracy))
							break
						else:
							logging.info('********** SWITCHING TO COUNTING, AVG POPULATION ACCURACY: %s' % str(average_accuracy))
							# we have to reset the accuracy before training a new task.
							for genome in genomes:
								genome.accuracy = 0.0
							generate(generations, population, all_possible_genes, dataset, 'count', mode_th, images_dir, stopping_th, epochs, debug_mode, genomes, evolver)
							already_switched= True
							#train_genomes(genomes, dataset, 'count', images_dir, epochs, debug_mode, mode_th)

				else:
					logging.info('********** SWITCHING TO COUNTING, ACCURACY: %s' % str(average_accuracy))
					#we have to reset the accuracy before training a new task.
					for genome in genomes:
						genome.accuracy = 0.0
					generate(generations, population, all_possible_genes, dataset, 'count', mode_th, images_dir, stopping_th, epochs, debug_mode, genomes, evolver)
					#train_genomes(genomes, dataset, 'count', images_dir, epochs, debug_mode, mode_th)
					already_switched = True
			else:
				train_genomes(genomes, dataset, 'size', images_dir, epochs, debug_mode, mode_th)
		elif mode == "count":
			train_genomes(genomes, dataset, 'count', images_dir, epochs, debug_mode, mode_th)
		else:
			train_genomes(genomes, dataset, 'size', images_dir, epochs, debug_mode, mode_th)
		# Print out the average accuracy each generation.
		average_accuracy = get_average_accuracy(genomes)
		logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
		logging.info('-'*80) #-----------

		# Evolve, except on the last iteration.
		if i != generations - 1:
			logging.info("Evolving!")
			genomes = evolver.evolve(genomes)

	# Sort our final population according to performance.
	genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

	# Print out the top 5 networks/genomes.
	print_genomes(genomes[:5])


def print_genomes(genomes):
	"""Print a list of genomes.

	Args:
		genomes (list): The population of networks/genomes

	"""
	logging.info('-'*80)

	for genome in genomes:
		genome.print_genome()


def analyze_data(images_path, analysis_path):
	df = pd.DataFrame()
	classes = []
	numbers = []
	convex_hulls_verteces = []
	convex_hulls_perimeter = []
	convex_hulls_area = []
	total_area = []
	#for i in range(0, 11):
#    file_names = os.listdir(images_path+'_'+str(i))
	file_names = os.listdir(images_path + '_0')
	for file_name in file_names:
		arr = file_name.split("_")
		class_name = arr[2]
		classes.append(class_name)
		number = arr[4]
		numbers.append(int(number))
		convex_hull_points = int(arr[6])
		convex_hulls_verteces.append(convex_hull_points)
		convex_hull_per = float(arr[8])
		convex_hulls_perimeter.append(convex_hull_per)
		convex_hull_area = float(arr[10])
		convex_hulls_area.append(convex_hull_area)
		area = arr[12]
		area = float(area[:area.find('png')-1])
		total_area.append(area)
	df['class'] = classes
	df['numeric_value'] = numbers
	df['convex_hull_verteces'] = convex_hulls_verteces
	df['convex_hull_perimeter'] = convex_hulls_perimeter
	df['convex_hull_area'] = convex_hulls_area
	df['total_area'] = total_area


	#histogram
	plt.figure(figsize=(8, 8))
	df['numeric_value'].hist(bins=70)
	title = 'Numeric value histogram in dataset'
	plt.title(title)
	plt.xlabel('numeric values')
	plt.ylabel('count')
	plt.savefig(analysis_path + os.sep + 'numeric_value.png')
	plt.close()

	#Count plot
	plt.figure(figsize=(8, 8))
	sns.countplot(x='class', data=df)
	title = 'Size count dataset: Count plot'
	plt.title(title)
	plt.savefig(analysis_path + os.sep + 'count_plot.png')
	plt.close()

	#pair plot
	plt.figure(figsize=(12, 12))
	sns.pairplot(data=df, hue='class')
	# title = 'Size count dataset: Pair plot'
	# plt.title(title)
	plt.savefig(analysis_path + os.sep + 'pair_plot.png')
	plt.close()

	#facet grid:
	plt.figure(figsize=(3, 12))
	g = sns.FacetGrid(data=df, col='class')
	g.map(plt.hist, 'numeric_value', bins=70)
	# title = 'Size count dataset: Facet grid histogram'
	# plt.title(title)
	plt.savefig(analysis_path + os.sep + 'facet_grid_hist.png')
	plt.close()

	df['class'] = df['class'].apply(convert_classes_to_numbers)

	#corr

	plt.figure(figsize = (12,12))
	sns.heatmap(df.corr(), cmap='coolwarm')
	plt.title('df.corr()')
	plt.savefig(analysis_path + os.sep + 'correlations.png')
	plt.close()


	print(df.head(10))


	return df


def convert_classes_to_numbers(class_name):
	if class_name == 'few':
		return 0
	else:
		return 1


def main(args):
	"""Evolve a genome."""
	population = args.population # Number of networks/genomes in each generation.
	#we only need to train the new ones....

	ds = args.ds

	if  (ds == 1):
		dataset = 'mnist_mlp'
	elif (ds == 2):
		dataset = 'mnist_cnn'
	elif (ds == 3):
		dataset = 'cifar10_mlp'
	elif (ds == 4):
		dataset = 'cifar10_cnn'
	elif (ds==5):
		dataset = 'size_count'
		#analyze_data(args.images_dir, args.analysis_path)
	else:
		dataset = 'mnist_mlp'

	print("***Dataset:", dataset)

	if dataset == 'mnist_cnn':
		generations = 8 # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [16, 32, 64, 128],
			'nb_layers':  [1, 2, 3, 4 ,5],
			'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
			'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
		}
	elif dataset == 'mnist_mlp':
		generations = 8 # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [64, 128], #, 256, 512, 768, 1024],
			'nb_layers':  [1, 2, 3, 4, 5],
			'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
			'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
		}
	elif dataset == 'cifar10_mlp':
		generations = 8 # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [64, 128, 256, 512, 768, 1024],
			'nb_layers':  [1, 2, 3, 4, 5],
			'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
			'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
		}
	elif dataset == 'cifar10_cnn':
		generations = 8 # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [16, 32, 64, 128],
			'nb_layers':  [1, 2, 3, 4, 5],
			'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
			'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
		}
	elif dataset == 'size_count':
		generations = args.gens  # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [16, 32, 64, 128],
			'nb_layers': [1, 2, 3, 4, 5],
			'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softplus', 'linear'],
			'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
		}
	else:
		generations = 8 # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [64, 128, 256, 512, 768, 1024],
			'nb_layers':  [1, 2, 3, 4, 5],
			'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
			'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
		}

	# replace nb_neurons with 1 unique value for each layer
	# 6th value reserved for dense layer
	nb_neurons = all_possible_genes['nb_neurons']
	for i in range(1, len(nb_neurons)+1):
	  all_possible_genes['nb_neurons_' + str(i)] = nb_neurons
	# remove old value from dict
	all_possible_genes.pop('nb_neurons')

	print("*** Evolving for %d generations with population size = %d ***" % (generations, population))

	generate(generations, population, all_possible_genes, dataset, args.mode, args.mode_th, args.images_dir, args.stopping_th, args.epochs, args.debug)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='evolve arguments')
	parser.add_argument('--datasource', dest='ds', type=int, required=True, help='The datasource')
	parser.add_argument('--population', dest='population', type=int, required=True, help='Number of networks/genomes in each generation.')
	parser.add_argument('--generations', dest='gens', type=int, required=True, help='Number of generations')
	parser.add_argument('--mode', dest='mode', type=str, required=True, help='task mode (size/count/both)')
	parser.add_argument('--mode_th', dest='mode_th', type=float, required=True, help='the mode threshold for moving from size to counting')
	parser.add_argument('--images_dir', dest='images_dir', type=str, required=True, help='The images dir')
	parser.add_argument('--stopping_th', dest='stopping_th', type=float, required=True, help='The stopping threshold of accuracy')
	parser.add_argument('--epochs', dest='epochs', type=int, required=True, help='The epochs')
	parser.add_argument('--debug', dest='debug', type=bool, required=False, default=False, help='debug')
	parser.add_argument('--analysis_path', dest='analysis_path', type=str, required=True, default='', help='analysis directory')

	args = parser.parse_args()
	main(args)

