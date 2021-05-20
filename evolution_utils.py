import pandas as pd
import numpy as np

#  RATIOS = [50,55,67,70,75,85] - these are the ratios if we cut after first point
RATIOS = [50, 56, 63, 71, 75, 85]
RATIO_NUMBERS = {50: [5, 10], 56: [5, 9], 63: [5, 8], 71: [5, 7], 75: [6, 8], 85: [6, 7]}

class DataPerSubject:
	def __init__(self, subject_uid,
				 training_accuracy,
				 validation_accuracy,
				 training_loss,
				 validation_loss,
				 training_congruency_result,
				 validation_congruency_result,
				 ratio_results):
		self.subject_uid = subject_uid
		self.training_accuracy = round(training_accuracy, 4)
		self.validation_accuracy = round(validation_accuracy, 4)
		self.training_loss = round(training_loss, 4)
		self.validation_loss = round(validation_loss, 4)
		self.training_congruency_result = training_congruency_result
		self.validation_congruency_result = validation_congruency_result
		self.ratio_results = ratio_results


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


def create_evolution_analysis_per_task_per_equate_csv(generation,
													  population,
													  data_from_all_subjects,
													  task,
													  equate,
													  training_set_size,
													  validation_set_size,
													  validation_set_size_congruent):
	evolution_analysis_result = []
	#for every generation
	headers = ['Subject',
			   'Subject_UID',
			   'Task',
			   'Equate',
			   'Generations',
			   'Training_Accuracy',
			   'Validation_Accuracy',
			   'Training_loss',
			   'Validation_Loss',
			   'Training_set_size',
			   'Validation_set_size',
			   'Validation_set_size_congruent',
			   'Validation_Accuracy_Congruent','Validation_Accuracy_Incongruent',
			   'Validation_Loss_Congruent','Validation_Loss_Incongruent',
			   'Training_Accuracy_Congruent', 'Training_Accuracy_Incongruent',
			   'Training_Loss_Congruent', 'Training_Loss_Incongruent',
			   'Ratio 50 Congruent Validation Accuracy','Ratio 50 Congruent Validation Loss',
			   'Ratio 50 Incongruent Validation Accuracy','Ratio 50 Incongruent Validation Loss',
			   'Ratio 55 Congruent Validation Accuracy','Ratio 55 Congruent Validation Loss',
			   'Ratio 55 Incongruent Validation Accuracy','Ratio 55 Incongruent Validation Loss',
			   'Ratio 65 Congruent Validation Accuracy','Ratio 65 Congruent Validation Loss',
			   'Ratio 65 Incongruent Validation Accuracy','Ratio 65 Incongruent Validation Loss',
			   'Ratio 75 Congruent Validation Accuracy','Ratio 75 Congruent Validation Loss',
			   'Ratio 75 Incongruent Validation Accuracy','Ratio 75 Incongruent Validation Loss',
			   'Ratio 85 Congruent Validation Accuracy','Ratio 85 Congruent Validation Loss',
			   'Ratio 85 Incongruent Validation Accuracy','Ratio 85 Incongruent Validation Loss']

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
			   validation_set_size_congruent,
			   data_from_all_subjects.validation_accuracies_congruent[subject],
			   data_from_all_subjects.validation_accuracies_incongruent[subject],
			   data_from_all_subjects.training_accuracies_congruent[subject],
			   data_from_all_subjects.training_accuracies_incongruent[subject]]

		ratios_dataset = data_from_all_subjects.ratio_results[subject]

		for ratio in RATIOS:
			row.append(round(ratios_dataset[ratio]["ratio_validation_accuracy_congruent"], 4))
			row.append(round(ratios_dataset[ratio]["ratio_validation_loss_congruent"], 4)),
			row.append(round(ratios_dataset[ratio]["ratio_validation_accuracy_incongruent"], 4)),
			row.append(round(ratios_dataset[ratio]["ratio_validation_loss_incongruent"], 4))

		evolution_analysis_result.append(row)

	df = pd.DataFrame(data=np.array(evolution_analysis_result), columns=headers)

	return df


def concat_dataframes_into_raw_data_csv_cross_generations(data_frame_list, file_name):
	result_df = pd.DataFrame()
	for df_per_gen in data_frame_list:
		result_df = pd.concat([result_df, df_per_gen], ignore_index=True)
	result_df.set_index('Subject', inplace=True)
	result_df.to_csv(file_name)