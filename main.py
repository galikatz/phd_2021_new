"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
from evolver import Evolver
from tqdm import tqdm
import logging
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from create_images_from_matlab import generate_new_images
import shutil

# Setup logging.
logging.basicConfig(
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%m/%d/%Y %I:%M:%S %p',
	level=logging.INFO#,
	#filename='log.txt'
)


def train_genomes(genomes, individuals_models, dataset, mode, path, epochs, debug_mode, mode_th):
	logging.info("*** Going to train %s individuals ***" % len(genomes))
	pop_size = len(genomes)
	#progress bar
	pbar = tqdm(total=pop_size)
	individual_index = 1
	best_individual_acc = 0.0
	best_individual_loss = 1.0
	sum_individual_acc = 0
	#loop over all individuals
	for genome in genomes:
		logging.info("*** Training individual #%s ***" % individual_index)
		if genome not in individuals_models:
			logging.info("*** Individual #%s is not in individuals_models, probably after evolution - new offspring ***" % individual_index)
			curr_individual_acc, curr_individual_loss, curr_individual_model = genome.train(dataset, mode, path, epochs,
																							debug_mode, mode_th,
																							best_individual_acc,
																							best_individual_loss,
																							None)

		else:
			logging.info("*** Individual #%s already in individuals_models ***" % individual_index)
			curr_individual_acc, curr_individual_loss, curr_individual_model = genome.train(dataset, mode, path, epochs,
																							debug_mode, mode_th,
																							best_individual_acc,
																							best_individual_loss,
																							individuals_models[genome])
		sum_individual_acc += curr_individual_acc

		individuals_models.update({genome: curr_individual_model})

		# finding the best individual in this generation
		if best_individual_acc < curr_individual_acc:
			best_individual_acc = curr_individual_acc
			best_individual_loss = curr_individual_loss
		elif best_individual_acc == curr_individual_acc and best_individual_loss > curr_individual_loss:
			best_individual_acc = curr_individual_acc
			best_individual_loss = curr_individual_loss

		pbar.update(1)
		individual_index += 1
	pbar.close()

	# calculate the avg accuracy in this generation
	avg_accuracy = sum_individual_acc / pop_size
	return best_individual_acc, best_individual_loss, individuals_models, avg_accuracy

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


def generate(generations, generation_index, population, all_possible_genes, dataset, mode, mode_th, images_dir,
			 stopping_th, epochs, debug_mode, congruency, equate, savedir, already_switched,
			 genomes=None, evolver=None, individual_models=None):
	"""Generate a network with the genetic algorithm.

	Args:
		generations (int): Number of times to evolve the population
		population (int): Number of networks in each generation
		all_possible_genes (dict): Parameter choices for networks
		dataset (str): Dataset to use for training/evaluating

	"""
	logging.info("*** Configuration: mode={}, mode_th={}, generations={}, population={}, epochs={}, stopping_th={})***".format(mode, mode_th, generations, population, epochs, stopping_th))

	if not genomes:
		evolver = Evolver(all_possible_genes)
		genomes = evolver.create_population(population)
		if not individual_models:
			individual_models = {}
		for genome in genomes:
			individual_models.update( {genome : None} )

	# Evolve the generation.
	if mode == 'both':
		actual_mode = 'size' #we start with size, than switch to counting
	else:
		actual_mode = mode
	for i in range(generation_index, generations + 1):
		### Every new generation we create new stimuli ###
		images_dir_per_gen = images_dir + "_" + str(i)

		if not os.path.exists(images_dir_per_gen):
			#delete old images
			if os.path.exists(images_dir + "_" + str(i-1)):
				shutil.rmtree(images_dir + "_" + str(i-1))
			# now generate the next dir
			generate_new_images(congruency, equate, savedir, i)

		logging.info("********* Now in mode %s generation %d of %d reading images from dir: %s *********" % (actual_mode, i, generations, images_dir_per_gen))

		print_genomes(genomes)

		# Train and Get the best accuracy for this generation from all individuals.
		# if there is no model existing for this genome it will create one.
		best_accuracy, best_loss, individuals_models, avg_accuracy = train_genomes(genomes, individual_models, dataset, actual_mode, images_dir_per_gen, epochs, debug_mode, mode_th)

		if mode != "both" and avg_accuracy >= stopping_th:
			logging.info("Done training! average_accuracy is %s" % str(avg_accuracy))
			break

		if mode == "both": # this is for the first time before the switch (no recursion)
			if avg_accuracy >= mode_th:
				if avg_accuracy >= stopping_th:
					if already_switched:
						logging.info("Done training! average_accuracy is %s" % str(avg_accuracy))
						break

				if not already_switched:
					logging.info('********** SWITCHING TO COUNTING, STILL IN GENERATION %s, ACCURACY: %s **********' % (str(i), str(best_accuracy)))
					actual_mode = 'count'
					# we have to reset the accuracy before training a new task.
					for genome in genomes:
						genome.accuracy = 0.0
						genome.val_loss = 1.0
					already_switched = True

				# now train again, this time for counting:
				best_accuracy, best_loss, individuals_models, avg_accuracy = train_genomes(genomes, individual_models, dataset, actual_mode, images_dir_per_gen, epochs, debug_mode, mode_th)
		# Print out the average accuracy each generation.
		logging.info("Generation avg accuracy: %.2f%%" % (avg_accuracy * 100))
		logging.info("Generation best accuracy: %.2f%% and loss: %.2f%%" % (best_accuracy * 100, best_loss))
		logging.info('-'*80) #-----------

		# Evolve, except on the last iteration.
		if i != generations:
			logging.info("Evolving! - mutation and recombination")
			genomes = evolver.evolve(genomes)

	logging.info("************ End of generations loop - evolution is over, avg accuracy: %.2f%%, best accuracy: %.2f%% and loss: %.2f%% **************" % (avg_accuracy * 100, best_accuracy * 100, best_loss))
	# Sort our final population according to performance.
	genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

	# Print out the top 5 networks/genomes.
	logging.info("Top 5 networks are:")
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
	if (ds==5):
		dataset = 'size_count'
		#analyze_data(args.images_dir, args.analysis_path)
	else:
		dataset = 'mnist_mlp'

	print("*** Dataset:", dataset)


	if dataset == 'size_count':
		generations = args.gens  # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [16, 32, 64, 128, 256],
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

	generate(generations, 1, population, all_possible_genes, dataset,
			 args.mode, args.mode_th, args.images_dir, args.stopping_th, args.epochs, args.debug, args.congruency,
			 args.equate, args.savedir, False)

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
	parser.add_argument('--congruency', dest='congruency', type=int, required=True, help='0-incongruent, 1-congruent')
	parser.add_argument('--equate', dest='equate', type=int, required=True,	help='1 is for average diameter; 2 is for total surface area; 3 is for convex hull')
	parser.add_argument('--savedir', dest='savedir', type=str, required=True, help='The save dir')
	args = parser.parse_args()
	main(args)

