import pandas as pd
import numpy as np

class DataPerSubject:
	def __init__(self, subject_uid,
				 training_accuracy,
				 validation_accuracy,
				 training_loss,
				 validation_loss,
				 validation_accuracy_congruent,
				 validation_accuracy_incongruent,
				 validation_loss_congruent,
				 validation_loss_incongruent):
		self.subject_uid = subject_uid
		self.training_accuracy = round(training_accuracy, 4)
		self.validation_accuracy = round(validation_accuracy, 4)
		self.training_loss = round(training_loss, 4)
		self.validation_loss = round(validation_loss, 4)
		self.validation_accuracy_congruent = round(validation_accuracy_congruent, 4)
		self.validation_accuracy_incongruent = round(validation_accuracy_incongruent, 4)
		self.validation_loss_congruent = round(validation_loss_congruent, 4)
		self.validation_loss_incongruent = round(validation_loss_incongruent, 4)

class DataAllSubjects:
	def __init__(self, dataPerSubjectList):
		self.subjects_uids = []
		self.training_accuracies = []
		self.validation_accuracies = []
		self.training_losses = []
		self.validation_losses = []
		self.validation_accuracies_congruent = []
		self.validation_accuracies_incongruent = []
		self.validation_losses_congruent = []
		self.validation_losses_incongruent = []

		for dataPerSubject in dataPerSubjectList:
			self.subjects_uids.append(dataPerSubject.subject_uid)
			self.training_accuracies.append(dataPerSubject.training_accuracy)
			self.validation_accuracies.append(dataPerSubject.validation_accuracy)
			self.training_losses.append(dataPerSubject.training_loss)
			self.validation_losses.append(dataPerSubject.validation_loss)
			self.validation_accuracies_congruent.append(dataPerSubject.validation_accuracy_congruent)
			self.validation_accuracies_incongruent.append(dataPerSubject.validation_accuracy_congruent)
			self.validation_losses_congruent.append(dataPerSubject.validation_loss_congruent)
			self.validation_losses_incongruent.append(dataPerSubject.validation_loss_incongruent)


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
			   'Validation_Accuracy_Congruent',
			   'Validation_Accuracy_Incongruent']

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
			   data_from_all_subjects.validation_accuracies_incongruent[subject]]
		evolution_analysis_result.append(row)

	df = pd.DataFrame(data=np.array(evolution_analysis_result), columns=headers)

	return df


def concat_dataframes_into_raw_data_csv_cross_generations(data_frame_list, file_name):
	result_df = pd.DataFrame()
	for df_per_gen in data_frame_list:
		result_df = pd.concat([result_df, df_per_gen], ignore_index=True)
	result_df.set_index('Subject', inplace=True)
	result_df.to_csv(file_name)