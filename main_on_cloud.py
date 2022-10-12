"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function

import time

import train
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
from sklearn.utils import shuffle
from train import refresh_classification_cache
from evolution_utils import create_evolution_analysis_per_task_per_equate_csv, \
	concat_dataframes_into_raw_data_csv_cross_generations, DataAllSubjects, evaluate_model
import glob
from evolution_utils import RATIOS
from datetime import datetime
import tensorflow as tf
from train_test_data import TrainGenomeResult
from load_networks_and_test import get_physical_properties_to_load, load_models
from classify import creating_train_test_data
from keras import losses
import json

MIN_DIFF = 100
NUM_OF_IMAGES_FILES = 20

# Setup logging.
logging.basicConfig(
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%m/%d/%Y %I:%M:%S %p',
	level=logging.INFO
)


def train_genomes(genomes, individuals_models, dataset, mode, equate, path, batch_size, epochs, debug_mode, training_strategy):
	logging.info("*** Going to train %s individuals ***" % len(genomes))
	pop_size = len(genomes)

	# progress bar
	pbar = tqdm(total=pop_size)
	individual_index = 1
	best_individual_acc = 0.0
	best_individual_loss = 1.0
	sum_individual_acc = 0

	# refresh classify cache
	logging.info("####################### Refreshing classification cache, once in a generation #######################")
	trainer_classification_cache = refresh_classification_cache()
	data_per_subject_list = []
	training_set_size = None
	validation_set_size = None
	validation_set_size_congruent = None

	############################
	# loop over all individuals
	# ##########################
	for genome in genomes:
		logging.info("*** Training individual #%s ***" % individual_index)

		if genome not in individuals_models:
			logging.info(
				"*** Individual #%s is not in individuals_models, probably after evolution - new offspring ***" % individual_index)
			train_result = genome.train(dataset, mode, equate, path, batch_size, epochs,
				debug_mode, best_individual_acc, None, trainer_classification_cache, training_strategy)
		else:
			logging.info("*** Individual #%s already in individuals_models ***" % individual_index)
			train_result = genome.train(dataset, mode, equate, path, batch_size, epochs,
				debug_mode, best_individual_acc, individuals_models[genome], trainer_classification_cache, training_strategy)

		sum_individual_acc += train_result.curr_individual_acc

		individuals_models.update({genome: train_result.curr_individual_model})

		# accumulate data per subject
		data_per_subject_list.append(train_result.data_per_subject)

		# finding the best individual in this generation
		if best_individual_acc < train_result.curr_individual_acc:
			best_individual_acc = train_result.curr_individual_acc
			best_individual_loss = train_result.curr_individual_loss
		elif best_individual_acc == train_result.curr_individual_acc and best_individual_loss > train_result.curr_individual_loss:
			best_individual_acc = train_result.curr_individual_acc
			best_individual_loss = train_result.curr_individual_loss

		pbar.update(1)
		individual_index += 1
	pbar.close()

	# calculate the avg accuracy in this generation
	avg_accuracy = sum_individual_acc / pop_size
	data_all_subjects = DataAllSubjects(data_per_subject_list)
	train_genome_result = TrainGenomeResult(best_individual_acc,
                 best_individual_loss,
                 individuals_models,
                 avg_accuracy,
                 data_all_subjects,
                 train_result.training_set_size,
                 train_result.validation_set_size,
                 train_result.validation_set_size_congruent)
	return train_genome_result


def get_best_genome(genomes):
	"""
	Gets the best individual in this generation
	:param genomes:
	:return: the accuracy score
	"""
	genomes_dict = {}
	max_accuracy = 0.0
	for genome in genomes:
		genomes_dict.update({genome.accuracy: genome})
		max_accuracy = max(max_accuracy, genome.accuracy)

	best_genome = genomes_dict.get(max_accuracy)
	logging.info("best genome has %f accuracy " % best_genome.accuracy)
	return best_genome


def generate(generations, generation_index, population, all_possible_genes, dataset, mode, mode_th, images_dir,
			 stopping_th, batch_size, epochs, debug_mode, congruency, equate, savedir, already_switched,
			 genomes=None, evolver=None, individual_models=None, should_delete_stimuli=False, running_on_cloud=False,
			 training_strategy=None, h5_path=None, should_train_first=True):
	"""Generate a network with the genetic algorithm.

	Args:
		generations (int): Number of times to evolve the population
		population (int): Number of networks in each generation
		all_possible_genes (dict): Parameter choices for networks
		dataset (str): Dataset to use for training/evaluating

	"""
	if should_train_first:
		logging.info(
			"*** Configuration: mode={}, mode_th={}, generations={}, population={}, batch_size={}.epochs={}, stopping_th={})***".format(
				mode, mode_th, generations, population, batch_size, epochs, stopping_th))

		if not genomes:
			evolver = Evolver(all_possible_genes)
			genomes = evolver.create_population(population)
			if not individual_models:
				individual_models = {}
			for genome in genomes:
				individual_models.update({genome: None})

		# Evolve the generation.
		if mode == 'size-count':
			actual_mode = 'size'  # we start with size, than switch to counting
		elif mode == 'colors-count':
			actual_mode = 'colors'
		elif mode == 'count-size':
			actual_mode = 'count'
		elif mode == 'colors':
			actual_mode = 'colors'
		else:
			actual_mode = mode

		dataframe_list_of_results = []

		########################
		# loop over generations
		# ######################
		start_time = time.time()
		i = 0
		avg_accuracy = None
		best_accuracy = None
		best_loss = None
		for i in range(generation_index, generations + 1):
			# Every new generation we create new stimuli, if there isn't we will modulu the generation number
			images_dir_per_gen = images_dir + "_" + str(i)
			if not os.path.isdir(images_dir_per_gen):
				images_dir_per_gen = images_dir + "_" + str(i % NUM_OF_IMAGES_FILES)
			if not running_on_cloud:
				creating_images_for_current_generation(images_dir_per_gen, images_dir, i, should_delete_stimuli, congruency,
													   equate, savedir, actual_mode, generations)
			balance(images_dir_per_gen)
			print_genomes(genomes)
			# Train and Get the best accuracy for this generation from all individuals.
			# if there is no model existing for this genome it will create one.
			train_genome_result = train_genomes(
				genomes, individual_models, dataset, actual_mode, equate, images_dir_per_gen, batch_size, epochs,
				debug_mode,
				training_strategy)

			avg_accuracy = train_genome_result.avg_accuracy
			best_accuracy = train_genome_result.best_individual_acc
			best_loss = train_genome_result.best_individual_loss

			if (mode != "size-count" and mode != "count-size" and mode != "colors-count") and avg_accuracy >= stopping_th:
				logging.info("Done training! average_accuracy is %s" % str(avg_accuracy))
				dataframe_list_of_results.append(
					accumulate_data(i, population, train_genome_result.data_all_subjects, mode, equate, train_genome_result.training_set_size,
									train_genome_result.validation_set_size, train_genome_result.validation_set_size_congruent))
				break

			if mode == "size-count" or mode == "colors-count" or mode == "count-size":  # this is for the first time before the switch
				if avg_accuracy >= mode_th:
					if avg_accuracy >= stopping_th:
						if already_switched:
							logging.info("Done training! average_accuracy is %s" % str(avg_accuracy))
							dataframe_list_of_results.append(
								accumulate_data(i, population, train_genome_result.data_all_subjects, mode, equate, train_genome_result.training_set_size,
												train_genome_result.validation_set_size, train_genome_result.validation_set_size_congruent))
							break

					if not already_switched:
						logging.info('********** SWITCHING TO 2nd task, STILL IN GENERATION %s, ACCURACY: %s **********' % (
							str(i), str(best_accuracy)))
						if mode == "size-count" or mode == 'colors-count':
							actual_mode = 'count'
						elif mode == "count-size":
							actual_mode = 'size'

						# we have to reset the accuracy before training a new task.
						for genome in genomes:
							genome.accuracy = 0.0
							genome.val_loss = 1.0
						already_switched = True

					# now train again, this time for counting:
					train_genome_result = train_genomes(
							genomes, individual_models, dataset, actual_mode, equate, images_dir_per_gen, batch_size,
							epochs,
							debug_mode, training_strategy)

					avg_accuracy = train_genome_result.avg_accuracy
					best_accuracy = train_genome_result.best_individual_acc
					best_loss = train_genome_result.best_individual_loss

			# Print out the average accuracy each generation.
			logging.info("Generation avg accuracy: %.2f%%" % (avg_accuracy * 100))
			logging.info("Generation best accuracy: %.2f%% and loss: %.2f%%" % (best_accuracy * 100, best_loss))
			logging.info('-' * 80)  # -----------

			# Evolve, except on the last iteration.
			if i != generations:
				logging.info("Evolving! - mutation and recombination")
				genomes = evolver.evolve(genomes)

			# create data for analysis per generation
			dataframe_list_of_results.append(
				accumulate_data(i, population, train_genome_result.data_all_subjects, mode, equate, train_genome_result.training_set_size, train_genome_result.validation_set_size,
								train_genome_result.validation_set_size_congruent))

		logging.info("************ End of generations loop - evolution is over, avg accuracy: %.2f%%, best accuracy: %.2f%% and loss: %.2f%% **************" % (
				avg_accuracy * 100, best_accuracy * 100, best_loss))
		# Sort our final population according to performance.
		genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

		# Print out the top 5 networks/genomes.
		logging.info("Top 5 networks are:")
		print_genomes(genomes[:5])

		# creating result csvs:
		logging.info("Creating results csvs")
		total_time = (time.time() - start_time) / 60
		now_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
		filename = "Results:%s_Mode:%s_Generations:%s_Population:%s_Equate:%s_Epochs:%s_AvgAccuracy:%.2f%%_Time:%s_minutes" % (
		now_str, mode, str(i), str(population), str(equate), str(epochs), avg_accuracy, str(round(total_time, 3)))
		filename = filename.replace(":", "_") + ".csv"
		concat_dataframes_into_raw_data_csv_cross_generations(dataframe_list_of_results, filename)
		#######################
		# save genomes to file
		#######################
		for genome in genomes:
			with open("models" + os.sep + "genome_" + str(genome.u_ID) + '.json', 'w') as f:
				json.dump(genome.__dict__, f)

		logging.info(f"Done training! took {total_time} minutes.")

	###############################################################
	# Testing on different stimuli only once when training is over!
	###############################################################

	logging.info("##########  Testing Loaded models %s #########")
	testing_loaded_models(h5_path, images_dir, equate, mode, population, batch_size)


def testing_loaded_models(h5_path, images_dir, equate, mode, population, batch_size):
	models_data = load_models(h5_path)
	list_of_stimuli_data = get_physical_properties_to_load(images_dir, equate)
	for stimuli_data in list_of_stimuli_data:
		new_train_test_data = creating_train_test_data(dir=stimuli_data.path, stimuli_type="katzin", mode=mode, nb_classes=train.FIXED_NB_CLASSES)
		avg_tested_acc = 0
		avg_tested_loss = 0
		test_dataframe_list_of_results = []
		test_data_all_subjects = []
		for loaded_genome in models_data:
			logging.info("##########  Testing model that was trained on equate_%s for genome: %s on %s #########" % (
			equate, str(loaded_genome.u_ID), stimuli_data.equate))

			optimizer = loaded_genome.geneparam['optimizer']

			model = models_data[loaded_genome]
			bce = losses.BinaryCrossentropy(reduction='none')
			model.compile(loss=bce, optimizer=optimizer, metrics=["accuracy"])

			test_genome_result = evaluate_model(genome=loaded_genome, model=model, history=None,
												train_test_data=new_train_test_data, batch_size=batch_size)
			test_data_all_subjects.append(test_genome_result.data_per_subject)

			avg_tested_acc += test_genome_result.curr_individual_acc
			avg_tested_loss = avg_tested_loss + test_genome_result.curr_individual_loss
		test_dataframe_list_of_results.append(
			accumulate_data(loaded_genome.generation, population, DataAllSubjects(test_data_all_subjects), mode, equate,
							test_genome_result.training_set_size, test_genome_result.validation_set_size,
							test_genome_result.validation_set_size_congruent))
		avg_tested_acc /= len(models_data)
		avg_tested_loss /= len(models_data)
		test_filename = "Results_%s_Mode_%s_Trained_on_Equate_%s_Tested_on_%s_AvgAccuracy_%.2f%%_AvgLoss_%.2f%%.csv" % (
			datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), mode, str(equate),
			stimuli_data.equate.replace("equate", "Equate"), avg_tested_acc, avg_tested_loss)
		concat_dataframes_into_raw_data_csv_cross_generations(test_dataframe_list_of_results, test_filename)
		logging.info(f"Done Testing individuals trained on Equate_%s tested on %s !" % (equate, stimuli_data.equate))


def accumulate_data(curr_gen, population, data_from_all_subjects, mode, equate, training_set_size, validation_set_size,
					validation_set_size_congruent):
	df_per_gen = create_evolution_analysis_per_task_per_equate_csv(curr_gen,
																   population,
																   data_from_all_subjects,
																   mode,
																   equate,
																   training_set_size,
																   validation_set_size,
																   validation_set_size_congruent)
	return df_per_gen


def creating_images_for_current_generation(images_dir_per_gen, images_dir, i, should_delete_stimuli, congruency, equate,
										   savedir, actual_mode, generations):
	total_num_of_files = 0
	total_num_of_cong = 0
	total_num_of_incong = 0
	if not os.path.exists(images_dir_per_gen):
		# delete old images
		if should_delete_stimuli and os.path.exists(images_dir + "_" + str(i - 1)):
			shutil.rmtree(images_dir + "_" + str(i - 1))
		# now generate the next dir
		if congruency == 2:  # both cong and incong are required
			for ratio in RATIOS:
				num_of_incong = generate_new_images(0, equate, savedir, i, "incong" + str(ratio), ratio, actual_mode)
				num_of_cong = generate_new_images(1, equate, savedir, i, "cong" + str(ratio), ratio, actual_mode)
				# balance incong and cong sizes:
				while num_of_cong - num_of_incong > MIN_DIFF:
					# create more incongruent
					num_of_incong = generate_new_images(0, equate, savedir, i, "incong" + str(ratio), ratio, actual_mode)

				while num_of_incong - num_of_cong > MIN_DIFF:
					# create more congruent
					num_of_cong = generate_new_images(1, equate, savedir, i, "cong" + str(ratio), ratio, actual_mode)
				# else they are equal - no need to create / delete anything

				# now balance per ratio
				if num_of_cong > num_of_incong:
					diff = num_of_cong - num_of_incong
					delete_extra_files("cong" + str(ratio), diff, images_dir_per_gen)
				if num_of_incong > num_of_cong:
					diff = num_of_incong - num_of_cong
					delete_extra_files("incong" + str(ratio), diff, images_dir_per_gen)

			# now balance the amount of files in all ratios to be the same.
			num_of_files_per_ratio = balance(images_dir_per_gen)

			logging.info("Data after deletion per ratio: %s" % num_of_files_per_ratio)
			total_num_of_cong += len(glob.glob(images_dir_per_gen + os.sep + 'cong*.jpg'))
			total_num_of_incong += len(glob.glob(images_dir_per_gen + os.sep + 'incong*.jpg'))
			total_num_of_files += (total_num_of_cong + total_num_of_incong)
		else:
		 	total_num_of_files = len(generate_new_images(congruency, equate, savedir, i, actual_mode))

		logging.info("Number of files created is: %s, incong: %s, cong: %s" % (
			total_num_of_files, total_num_of_incong, total_num_of_cong))
	logging.info("********* Now in mode %s generation %d out of %d reading images from dir: %s *********" % (
		actual_mode, i, generations, images_dir_per_gen))


def validate_balanced_stimuli(images_dir_per_gen) -> ({}, bool):
	num_of_files_per_ratio = {}
	is_valid = True
	for ratio in RATIOS:
		num_of_cong = len(glob.glob(images_dir_per_gen + os.sep + 'cong' + str(ratio) + '*.jpg'))
		num_of_incong = len(glob.glob(images_dir_per_gen + os.sep + 'incong' + str(ratio) + '*.jpg'))
		# now balance by congruency
		if num_of_cong > num_of_incong:
			is_valid = False

		if num_of_incong > num_of_cong:
			is_valid = False

		num_of_files_per_ratio.update({ratio: (num_of_incong + num_of_cong)})

	min_num_of_files = 100000
	for ratio in RATIOS:
		if min_num_of_files > num_of_files_per_ratio[ratio]:
			min_num_of_files = num_of_files_per_ratio[ratio]
	for ratio in RATIOS:
		if min_num_of_files != num_of_files_per_ratio[ratio]:
			is_valid = False
	return (num_of_files_per_ratio, is_valid)


def balance(images_dir_per_gen):
	(num_of_files_per_ratio, is_valid) = validate_balanced_stimuli(images_dir_per_gen)
	if is_valid:
		logging.info("Data is balanced: %s" % num_of_files_per_ratio)
		return

	num_of_files_per_ratio = {}
	for ratio in RATIOS:
		num_of_cong = len(glob.glob(images_dir_per_gen + os.sep + 'cong' + str(ratio) + '*.jpg'))
		num_of_incong = len(glob.glob(images_dir_per_gen + os.sep + 'incong' + str(ratio) + '*.jpg'))
		# now balance by congruency
		if num_of_cong > num_of_incong:
			diff = num_of_cong - num_of_incong
			delete_extra_files("cong" + str(ratio), diff, images_dir_per_gen)
		if num_of_incong > num_of_cong:
			diff = num_of_incong - num_of_cong
			delete_extra_files("incong" + str(ratio), diff, images_dir_per_gen)

		num_of_files_per_ratio.update({ratio: (num_of_incong + num_of_cong)})

	min_num_of_files = 100000
	min_ration = -1
	for ratio in RATIOS:
		if min_num_of_files > num_of_files_per_ratio[ratio]:
			min_num_of_files = num_of_files_per_ratio[ratio]
			min_ration = ratio
	logging.info("Minimum number is: %s for ratio: %s" % (min_num_of_files, min_ration))
	logging.info("Original data before deletion per ratio: %s" % num_of_files_per_ratio)
	for ratio in RATIOS:
		if ratio == min_ration:
			continue
		diff = num_of_files_per_ratio[ratio] - min_num_of_files
		half_diff = int(diff / 2)
		logging.info("Going to delete : %s files from ratio %s, half %s cong, original was: %s" % (
			diff, ratio, half_diff, num_of_files_per_ratio[ratio]))
		delete_extra_files("incong" + str(ratio), half_diff, images_dir_per_gen)
		delete_extra_files("cong" + str(ratio), half_diff, images_dir_per_gen)
		num_of_incong = len(glob.glob(images_dir_per_gen + os.sep + 'incong' + str(ratio) + '*.jpg'))
		num_of_cong = len(glob.glob(images_dir_per_gen + os.sep + 'cong' + str(ratio) + '*.jpg'))
		num_of_files_per_ratio.update({ratio: (num_of_incong + num_of_cong)})
		logging.info("Number of files per ratio: %s is: incong: %s, cong: %s" % (ratio, num_of_incong, num_of_cong))
	logging.info("Data after deletion per ratio: %s" % num_of_files_per_ratio)
	return num_of_files_per_ratio


def delete_extra_files(prefix, num_of_files_to_delete, images_dir):
	images_files = os.listdir(images_dir)
	images_files_shuffled = shuffle(images_files)
	files_to_delete = []
	for file_name in images_files_shuffled:
		if file_name.startswith(prefix):
			files_to_delete.append(file_name)
		if len(files_to_delete) == num_of_files_to_delete:
			break
	for fname in files_to_delete:
		os.remove(os.path.join(images_dir, fname))
	logging.info("Done deleting %s files, for balancing stimuli" % len(files_to_delete))


def print_genomes(genomes):
	"""Print a list of genomes.

	Args:
		genomes (list): The population of networks/genomes

	"""
	logging.info('-' * 80)

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
	# for i in range(0, 11):
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
		area = float(area[:area.find('png') - 1])
		total_area.append(area)
	df['class'] = classes
	df['numeric_value'] = numbers
	df['convex_hull_verteces'] = convex_hulls_verteces
	df['convex_hull_perimeter'] = convex_hulls_perimeter
	df['convex_hull_area'] = convex_hulls_area
	df['total_area'] = total_area

	# histogram
	plt.figure(figsize=(8, 8))
	df['numeric_value'].hist(bins=70)
	title = 'Numeric value histogram in dataset'
	plt.title(title)
	plt.xlabel('numeric values')
	plt.ylabel('count')
	plt.savefig(analysis_path + os.sep + 'numeric_value.png')
	plt.close()

	# Count plot
	plt.figure(figsize=(8, 8))
	sns.countplot(x='class', data=df)
	title = 'Size count dataset: Count plot'
	plt.title(title)
	plt.savefig(analysis_path + os.sep + 'count_plot.png')
	plt.close()

	# pair plot
	plt.figure(figsize=(12, 12))
	sns.pairplot(data=df, hue='class')
	# title = 'Size count dataset: Pair plot'
	# plt.title(title)
	plt.savefig(analysis_path + os.sep + 'pair_plot.png')
	plt.close()

	# facet grid:
	plt.figure(figsize=(3, 12))
	g = sns.FacetGrid(data=df, col='class')
	g.map(plt.hist, 'numeric_value', bins=70)
	# title = 'Size count dataset: Facet grid histogram'
	# plt.title(title)
	plt.savefig(analysis_path + os.sep + 'facet_grid_hist.png')
	plt.close()

	df['class'] = df['class'].apply(convert_classes_to_numbers)
	# corr
	plt.figure(figsize=(12, 12))
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


def create_tpu_strategy():
	tpu_strategy = None
	try:
		tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
		print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
		tf.config.experimental_connect_to_cluster(tpu)
		tf.tpu.experimental.initialize_tpu_system(tpu)
		tpu_strategy = tf.distribute.TPUStrategy(tpu)
	except ValueError:
		logging.error('ERROR: Not connected to a TPU runtime')
	return tpu_strategy


def create_gpu_strategy():
	return tf.distribute.MirroredStrategy()


def main(args):
	training_strategy = None
	if args.strategy == "TPU":
		training_strategy = create_tpu_strategy()
	elif args.strategy == "GPU":
		logging.info('Running on GPU')
		training_strategy = create_gpu_strategy()

	"""Evolve a genome."""
	population = args.population  # Number of networks/genomes in each generation.
	# we only need to train the new ones....
	if args.ds == 5:
		dataset = 'size_count'
	# analyze_data(args.images_dir, args.analysis_path)
	else:
		dataset = 'mnist_mlp'

	print("*** Dataset:", dataset)

	if dataset == 'size_count':
		generations = args.gens  # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [16, 32, 64, 128],
			#  'nb_neurons': [16, 32, 64, 128, 256],
			'nb_layers': [2, 3, 4, 5],
			'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softplus', 'linear'],
			'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
		}
	else:
		generations = 8  # Number of times to evolve the population.
		all_possible_genes = {
			'nb_neurons': [64, 128, 256, 512, 768, 1024],
			'nb_layers': [1, 2, 3, 4, 5],
			'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softplus', 'linear'],
			'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
		}

	# replace nb_neurons with 1 unique value for each layer
	# 6th value reserved for dense layer
	nb_neurons = all_possible_genes['nb_neurons']
	for i in range(1, len(nb_neurons) + 1):
		all_possible_genes['nb_neurons_' + str(i)] = nb_neurons
	# remove old value from dict
	all_possible_genes.pop('nb_neurons')

	print("*** Evolving for %d generations with population size = %d ***" % (generations, population))
	batch_size = args.batch_size
	if args.strategy == "TPU":
		batch_size = 8 * training_strategy.num_replicas_in_sync
		logging.info("*** According to TPU strategy Batch size is %s ***" % batch_size)
		if args.mode == 'count':  # smaller batch because of OOM
			batch_size = 16
			logging.info("*** Batch size was fixed for mode: %s to: %s ***" % (args.mode, batch_size))

	generate(generations=generations, generation_index=1, population=population, all_possible_genes=all_possible_genes,
			 dataset=dataset,
			 mode=args.mode, mode_th=args.mode_th, images_dir=args.images_dir, stopping_th=args.stopping_th,
			 batch_size=batch_size, epochs=args.epochs, debug_mode=args.debug, congruency=args.congruency,
			 equate=args.equate, savedir=args.savedir, already_switched=False,
			 genomes=None, evolver=None, individual_models=None, should_delete_stimuli=args.should_delete_stimuli,
			 running_on_cloud=args.running_on_cloud, training_strategy=training_strategy, h5_path=args.h5_path, should_train_first=args.should_train_first)


def str2bool(value):
	"""Convert string to bool (in argparse context)."""
	if value.lower() not in ['true', 'false', '1', '0']:
		raise ValueError('Need bool; got %r' % value)
	return {'true': True, 'false': False, '1': True, '0': False}[value.lower()]


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='evolve arguments')
	parser.add_argument('--datasource', dest='ds', type=int, required=True, help='The datasource')
	parser.add_argument('--population', dest='population', type=int, required=True,
						help='Number of networks/genomes in each generation.')
	parser.add_argument('--generations', dest='gens', type=int, required=True, help='Number of generations')
	parser.add_argument('--mode', dest='mode', type=str, required=True, help='task mode (size/count/both)')
	parser.add_argument('--mode_th', dest='mode_th', type=float, required=True,
						help='the mode threshold for moving from size to counting')
	parser.add_argument('--images_dir', dest='images_dir', type=str, required=True, help='The images dir')
	parser.add_argument('--stopping_th', dest='stopping_th', type=float, required=True,
						help='The stopping threshold of accuracy')
	parser.add_argument('--epochs', dest='epochs', type=int, required=True, help='The epochs')
	parser.add_argument('--debug', dest='debug', type=str2bool, required=False, default=False, help='debug')
	parser.add_argument('--analysis_path', dest='analysis_path', type=str, required=True, default='',
						help='analysis directory')
	parser.add_argument('--congruency', dest='congruency', type=int, required=True,
						help='0-incongruent, 1-congruent, 2-both')
	parser.add_argument('--equate', dest='equate', type=int, required=True,
						help='1 is for average diameter; 2 is for total surface area; 3 is for convex hull')
	parser.add_argument('--savedir', dest='savedir', type=str, required=True, help='The save dir')
	parser.add_argument('--should_delete_stimuli', dest='should_delete_stimuli', type=str2bool, required=False,
						default=False, help='should delete old generations stimuli images dir')
	parser.add_argument('--batch_size', dest='batch_size', type=int, required=True, help='The batch_size')
	parser.add_argument('--running_on_cloud', dest='running_on_cloud', type=str2bool, required=False,
						help='running on a cloud or locally', default=False)
	parser.add_argument('--strategy', dest='strategy', type=str, required=False, help='Running on cloud GPU/TPU/CPU',
						default="CPU")
	parser.add_argument('--h5_path', dest='h5_path', type=str, required=False, help='h5_path',default="/Users/gali.k/phd/phd_2021/models")
	parser.add_argument('--should_train_first', dest='should_train_first', type=str2bool, required=False, default=True, help='should train first or skip to test')
	args = parser.parse_args()
	main(args)
