import pandas as pd
import numpy as np
import os
from train_test_data import TrainResult
from sklearn.metrics import roc_auc_score

RATIOS = [50, 56, 63, 71, 75, 86]
RATIO_NUMBERS = {50: [5, 10], 56: [5, 9], 63: [5, 8], 71: [5, 7], 75: [6, 8], 86: [6, 7]}


class DataPerSubject:
	def __init__(self, subject_uid,
				 training_accuracy,
				 validation_accuracy,
				 training_loss,
				 validation_loss,
				 training_congruency_result,
				 validation_congruency_result,
				 ratio_results,
				 nb_layers,
				 nb_neurons,
				 activation,
				 optimizer
	):
		self.subject_uid = subject_uid
		self.training_accuracy = round(training_accuracy, 4)
		self.validation_accuracy = round(validation_accuracy, 4)
		self.training_loss = round(training_loss, 4)
		self.validation_loss = round(validation_loss, 4)
		self.training_congruency_result = training_congruency_result
		self.validation_congruency_result = validation_congruency_result
		self.ratio_results = ratio_results
		self.nb_layers = nb_layers
		self.nb_neurons = nb_neurons
		self.activation = activation
		self.optimizer = optimizer


class DataAllSubjects:
	def __init__(self, dataPerSubjectList):
		self.subjects_uids = []
		self.training_accuracies = []
		self.validation_accuracies = []
		self.training_losses = []
		self.validation_losses = []
		self.training_congruency_result = []
		self.validation_congruency_result = []
		self.ratio_results = []
		self.nb_layers = []
		self.nb_neurons = []
		self.activation = []
		self.optimizer = []

		for dataPerSubject in dataPerSubjectList:
			self.subjects_uids.append(dataPerSubject.subject_uid)
			#  main data
			self.training_accuracies.append(dataPerSubject.training_accuracy)
			self.validation_accuracies.append(dataPerSubject.validation_accuracy)
			self.training_losses.append(dataPerSubject.training_loss)
			self.validation_losses.append(dataPerSubject.validation_loss)

			#  congruency data
			self.training_congruency_result.append(dataPerSubject.training_congruency_result)
			self.validation_congruency_result.append(dataPerSubject.validation_congruency_result)

			# ratio data
			self.ratio_results.append(dataPerSubject.ratio_results)

			# chosen architecture params
			self.nb_layers.append(dataPerSubject.nb_layers)
			self.nb_neurons.append(dataPerSubject.nb_neurons)
			self.activation.append(dataPerSubject.activation)
			self.optimizer.append(dataPerSubject.optimizer)


def create_evolution_analysis_per_task_per_equate_csv(generation,
													  population,
													  data_from_all_subjects,
													  task,
													  equate,
													  training_set_size,
													  validation_set_size,
													  validation_set_size_congruent):
	evolution_analysis_result = []
	# for every generation
	headers = ['Subject',
			   'Subject_UID',
			   'Task',
			   'Equate',
			   'Generations',
			   'Training_Accuracy',
			   'Validation_Accuracy',
			   'Training_Loss',
			   'Validation_Loss',
			   'Training_set_size',
			   'Validation_set_size',
			   'Validation_set_size_congruent',

			   'Ratio 50 Congruent Training Accuracy',
			   'Ratio 50 Incongruent Training Accuracy',
			   'Ratio 50 Congruent Training Loss',
			   'Ratio 50 Incongruent Training Loss',

			   'Ratio 50 Congruent Validation Accuracy',
			   'Ratio 50 Incongruent Validation Accuracy',
			   'Ratio 50 Congruent Validation Loss',
			   'Ratio 50 Incongruent Validation Loss',

			   'Ratio 56 Congruent Training Accuracy',
			   'Ratio 56 Incongruent Training Accuracy',
			   'Ratio 56 Congruent Training Loss',
			   'Ratio 56 Incongruent Training Loss',

			   'Ratio 56 Congruent Validation Accuracy',
			   'Ratio 56 Incongruent Validation Accuracy',
			   'Ratio 56 Congruent Validation Loss',
			   'Ratio 56 Incongruent Validation Loss',

			   'Ratio 63 Congruent Training Accuracy',
			   'Ratio 63 Incongruent Training Accuracy',
			   'Ratio 63 Congruent Training Loss',
			   'Ratio 63 Incongruent Training Loss',

			   'Ratio 63 Congruent Validation Accuracy',
			   'Ratio 63 Incongruent Validation Accuracy',
			   'Ratio 63 Congruent Validation Loss',
			   'Ratio 63 Incongruent Validation Loss',

			   'Ratio 71 Congruent Training Accuracy',
			   'Ratio 71 Incongruent Training Accuracy',
			   'Ratio 71 Congruent Training Loss',
			   'Ratio 71 Incongruent Training Loss',

			   'Ratio 71 Congruent Validation Accuracy',
			   'Ratio 71 Incongruent Validation Accuracy',
			   'Ratio 71 Congruent Validation Loss',
			   'Ratio 71 Incongruent Validation Loss',

			   'Ratio 75 Congruent Training Accuracy',
			   'Ratio 75 Incongruent Training Accuracy',
			   'Ratio 75 Congruent Training Loss',
			   'Ratio 75 Incongruent Training Loss',

			   'Ratio 75 Congruent Validation Accuracy',
			   'Ratio 75 Incongruent Validation Accuracy',
			   'Ratio 75 Congruent Validation Loss',
			   'Ratio 75 Incongruent Validation Loss',

			   'Ratio 86 Congruent Training Accuracy',
			   'Ratio 86 Incongruent Training Accuracy',
			   'Ratio 86 Congruent Training Loss',
			   'Ratio 86 Incongruent Training Loss',

			   'Ratio 86 Congruent Validation Accuracy',
			   'Ratio 86 Incongruent Validation Accuracy',
			   'Ratio 86 Congruent Validation Loss',
			   'Ratio 86 Incongruent Validation Loss',

			   'Layers',
			   'Nuerons',
			   'Activation',
			   'Optimizer']

	for subject in range (0, population):

		row = [subject+1,
			   data_from_all_subjects.subjects_uids[subject],
			   task,
			   equate,
			   generation,
			   data_from_all_subjects.training_accuracies[subject],
			   data_from_all_subjects.validation_accuracies[subject],
			   data_from_all_subjects.training_losses[subject],
			   data_from_all_subjects.validation_losses[subject],
			   training_set_size,
			   validation_set_size,
			   validation_set_size_congruent]

		ratios_dataset = data_from_all_subjects.ratio_results[subject]

		for ratio in RATIOS:
			row.append(round(ratios_dataset[ratio][0]["ratio_training_accuracy_congruent"], 4))
			row.append(round(ratios_dataset[ratio][1]["ratio_training_accuracy_incongruent"], 4)),
			row.append(round(ratios_dataset[ratio][2]["ratio_training_loss_congruent"], 4)),
			row.append(round(ratios_dataset[ratio][3]["ratio_training_loss_incongruent"], 4))

			row.append(round(ratios_dataset[ratio][4]["ratio_validation_accuracy_congruent"], 4))
			row.append(round(ratios_dataset[ratio][5]["ratio_validation_accuracy_incongruent"], 4)),
			row.append(round(ratios_dataset[ratio][6]["ratio_validation_loss_congruent"], 4)),
			row.append(round(ratios_dataset[ratio][7]["ratio_validation_loss_incongruent"], 4))

		row.append(data_from_all_subjects.nb_layers[subject])
		row.append(data_from_all_subjects.nb_neurons[subject])
		row.append(data_from_all_subjects.activation[subject])
		row.append(data_from_all_subjects.optimizer[subject])

		evolution_analysis_result.append(row)

	df = pd.DataFrame(data=np.array(evolution_analysis_result, dtype=object), columns=headers)

	return df


def concat_dataframes_into_raw_data_csv_cross_generations(data_frame_list, file_name):
	result_df = pd.DataFrame()
	for df_per_gen in data_frame_list:
		result_df = pd.concat([result_df, df_per_gen], ignore_index=True)
	result_df.set_index('Subject', inplace=True)
	result_df.to_csv("results" + os.sep + file_name)


def evaluate_model(genome, model, history, train_test_data, batch_size):
	# The score contains the validation accuracy (score[1]) and the validation loss score[0])
	score = model.evaluate(x=train_test_data.x_test, y=train_test_data.y_test, batch_size=batch_size, verbose=0)

	# # taking the last epoch result to be kept ( and not all the loss and accuracies from all epochs, since the last epoch is the best)
	if history is not None:
		training_accuracy = history.history["accuracy"][-1]
		validation_accuracy = history.history["val_accuracy"][-1]
		training_loss = history.history["loss"][-1]
		validation_loss = history.history["val_loss"][-1]
	else: # when we are in testing other stimuli we don't have history of training
		train_score = model.evaluate(x=train_test_data.x_train, y=train_test_data.y_train, batch_size=batch_size, verbose=0)
		training_accuracy = train_score[1]
		validation_accuracy = score[1]
		training_loss = train_score[0]
		validation_loss = score[0]

	# evaluate training congruency
	training_score_congruent = model.evaluate(x=train_test_data.x_cong_train, y=train_test_data.y_cong_train,
											  batch_size=batch_size, verbose=0)
	training_score_incongruent = model.evaluate(x=train_test_data.x_incong_train, y=train_test_data.y_incong_train,
												batch_size=batch_size, verbose=0)

	training_accuracy_congruent = training_score_congruent[1]
	training_accuracy_incongruent = training_score_incongruent[1]
	training_loss_congruent = training_score_congruent[0]
	training_loss_incongruent = training_score_incongruent[0]

	# evaluate validation congruency
	validation_score_congruent = model.evaluate(x=train_test_data.x_cong_test, y=train_test_data.y_cong_test,
												batch_size=batch_size, verbose=0)
	validation_score_incongruent = model.evaluate(x=train_test_data.x_incong_test, y=train_test_data.y_incong_test,
												  batch_size=batch_size, verbose=0)

	validation_accuracy_congruent = validation_score_congruent[1]
	validation_accuracy_incongruent = validation_score_incongruent[1]
	validation_loss_congruent = validation_score_congruent[0]
	validation_loss_incongruent = validation_score_incongruent[0]

	training_congruency_result = {"training_accuracy_congruent": training_accuracy_congruent,
								  "training_accuracy_incongruent": training_accuracy_incongruent,
								  "training_loss_congruent": training_loss_congruent,
								  "training_loss_incongruent": training_loss_incongruent}
	validation_congruency_result = {"validation_accuracy_congruent": validation_accuracy_congruent,
									"validation_accuracy_incongruent": validation_accuracy_incongruent,
									"validation_loss_congruent": validation_loss_congruent,
									"validation_loss_incongruent": validation_loss_incongruent}

	ratio_results = {}
	for ratio in train_test_data.ratios_validation_dataset:
		training_cong_touple = train_test_data.ratios_validation_dataset[ratio][0]
		training_incong_touple = train_test_data.ratios_validation_dataset[ratio][1]
		x_ratio_cong_train = training_cong_touple[0]
		y_ratio_cong_train = training_cong_touple[1]
		x_ratio_incong_train = training_incong_touple[0]
		y_ratio_incong_train = training_incong_touple[1]

		validation_cong_touple = train_test_data.ratios_validation_dataset[ratio][0]
		validation_incong_touple = train_test_data.ratios_validation_dataset[ratio][1]
		x_ratio_cong_test = validation_cong_touple[0]
		y_ratio_cong_test = validation_cong_touple[1]
		x_ratio_incong_test = validation_incong_touple[0]
		y_ratio_incong_test = validation_incong_touple[1]

		training_score_ratio_congruent = model.evaluate(x=x_ratio_cong_train, y=y_ratio_cong_train,
														batch_size=batch_size, verbose=0)
		training_score_ratio_incongruent = model.evaluate(x=x_ratio_incong_train, y=y_ratio_incong_train,
														  batch_size=batch_size, verbose=0)

		vaildation_score_ratio_congruent = model.evaluate(x=x_ratio_cong_test, y=y_ratio_cong_test,
														  batch_size=batch_size, verbose=0)
		vaildation_score_ratio_incongruent = model.evaluate(x=x_ratio_incong_test, y=y_ratio_incong_test,
															batch_size=batch_size, verbose=0)

		ratio_training_accuracy_congruent = training_score_ratio_congruent[1]
		ratio_training_accuracy_incongruent = training_score_ratio_incongruent[1]
		ratio_training_loss_congruent = training_score_ratio_congruent[0]
		ratio_training_loss_incongruent = training_score_ratio_incongruent[0]

		ratio_validation_accuracy_congruent = vaildation_score_ratio_congruent[1]
		ratio_validation_accuracy_incongruent = vaildation_score_ratio_incongruent[1]
		ratio_validation_loss_congruent = vaildation_score_ratio_congruent[0]
		ratio_validation_loss_incongruent = vaildation_score_ratio_incongruent[0]
		ratio_results.update({ratio: [{"ratio_training_accuracy_congruent": ratio_training_accuracy_congruent},
									  {"ratio_training_accuracy_incongruent": ratio_training_accuracy_incongruent},
									  {"ratio_training_loss_congruent": ratio_training_loss_congruent},
									  {"ratio_training_loss_incongruent": ratio_training_loss_incongruent},
									  {"ratio_validation_accuracy_congruent": ratio_validation_accuracy_congruent},
									  {"ratio_validation_accuracy_incongruent": ratio_validation_accuracy_incongruent},
									  {"ratio_validation_loss_congruent": ratio_validation_loss_congruent},
									  {"ratio_validation_loss_incongruent": ratio_validation_loss_incongruent}]})

	data_per_subject = DataPerSubject(genome.u_ID,
									  training_accuracy,
									  validation_accuracy,
									  training_loss,
									  validation_loss,
									  training_congruency_result,
									  validation_congruency_result,
									  ratio_results,
									  genome.geneparam['nb_layers'],
									  genome.nb_neurons(),
									  genome.geneparam['activation'],
									  genome.geneparam['optimizer'])

	training_set_size = len(train_test_data.x_train)
	validation_set_size = len(train_test_data.x_test)
	validation_set_size_congruent = len(train_test_data.x_cong_test)

	roc_score, y_test_corrected = predict_and_calc_roc_score(model, train_test_data, batch_size)

	best_current_val_loss = round(score[0], 3)
	best_current_val_accuracy = round(score[1], 3)
	print('Best current test loss from all epochs:',
		  best_current_val_loss)  # takes the minimum loss from all the epochs
	print('Best current test accuracy from all epochs based on minimal loss:',
		  best_current_val_accuracy)  # taking the accuracy of the minimal loss above.

	train_result = TrainResult(best_current_val_accuracy,
							   best_current_val_loss,
							   y_test_corrected,
							   model,
							   data_per_subject,
							   training_set_size,
							   validation_set_size,
							   validation_set_size_congruent)
	return train_result


def predict_and_calc_roc_score(model, train_test_data, batch_size):
	# saving the results of each prediction
	y_test_prediction = model.predict(x=train_test_data.x_test, batch_size=batch_size, verbose=0)

	# fixing the prediction result to be 0 and 1 and not float thresholds.
	y_test_corrected = []
	for i in range(len(y_test_prediction)):
		if y_test_prediction[i][0] > 0.5:
			left_stimulus_result = 1
			right_stimulus_result = 0
		else:
			left_stimulus_result = 0
			right_stimulus_result = 1
		y_test_corrected.append(np.array([left_stimulus_result, right_stimulus_result]))
	roc_score = roc_auc_score(train_test_data.y_test, y_test_corrected)
	return roc_score, y_test_corrected