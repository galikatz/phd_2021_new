import pandas as pd
import numpy as np
def create_evolution_analysis_per_task_per_equate_csv(generation,
					 population_size,
					 task,
					 equate,
					 training_accuracy_list,
					 validation_accuracy_list,
					 training_loss_list,
					 validation_loss_list,
					 training_set_size,
					 validation_set_size,
					 validation_set_size_congruent,
					 validation_accuracy_congruent_list,
					 validation_accuracy_incongruent_list):
	evolution_analysis_result = []
	#for every generation
	headers = ['Subject',
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

	for subject in range (0, population_size):
		row = [subject+1,
			   task,
			   equate,
			   generation,
			   training_accuracy_list[subject],
			   validation_accuracy_list[subject],
			   training_loss_list[subject],
			   validation_loss_list[subject],
			   training_set_size,
			   validation_set_size,
			   validation_set_size_congruent,
			   validation_accuracy_congruent_list[subject],
			   validation_accuracy_incongruent_list[subject]]
		evolution_analysis_result.append(row)

	df = pd.DataFrame(data=np.array(evolution_analysis_result), columns=headers)

	return df

if __name__ == '__main__':
	df_gen_1 = create_evolution_analysis_per_task_per_equate_csv(
					 1,
					 4,
					'size',
						3,
					 [0.6,0.8,0.9,0.6],
					 [0.8, 0.9, 0.9, 0.7],
					 [0.1, 0.02, 0.03, 0.014],
					 [0.2, 0.01, 0.02, 0.016],
					 730,
					 200,
					 100,
					 [0.8,0.8,0.9,0.7],
					 [0.6, 0.7, 0.8, 0.7])



	df_gen_2 = create_evolution_analysis_per_task_per_equate_csv(
		1,
		4,
		'size',
		1,
		[0.6, 0.8, 0.9, 0.6],
		[0.8, 0.9, 0.9, 0.7],
		[0.1, 0.02, 0.03, 0.014],
		[0.2, 0.01, 0.02, 0.016],
		730,
		200,
		100,
		[0.8, 0.8, 0.9, 0.7],
		[0.6, 0.7, 0.8, 0.7])

	df_gen_3 = create_evolution_analysis_per_task_per_equate_csv(
		2,
		4,
		'count',
		3,
		[0.6, 0.8, 0.9, 0.6],
		[0.8, 0.9, 0.9, 0.7],
		[0.1, 0.02, 0.03, 0.014],
		[0.2, 0.01, 0.02, 0.016],
		730,
		200,
		100,
		[0.8, 0.8, 0.9, 0.7],
		[0.6, 0.7, 0.8, 0.7])

	df_gen_4 = create_evolution_analysis_per_task_per_equate_csv(
		2,
		4,
		'count',
		2,
		[0.6, 0.8, 0.9, 0.6],
		[0.8, 0.9, 0.9, 0.7],
		[0.1, 0.02, 0.03, 0.014],
		[0.2, 0.01, 0.02, 0.016],
		730,
		200,
		100,
		[0.8, 0.8, 0.9, 0.7],
		[0.6, 0.7, 0.8, 0.7])

	result_df = pd.concat([df_gen_1, df_gen_2])
	result_df = pd.concat([result_df, df_gen_3], ignore_index=True)
	result_df = pd.concat([result_df, df_gen_4], ignore_index=True)

	result_df.set_index('Subject', inplace=True)
	result_df.to_csv("evolution_analysis.csv")