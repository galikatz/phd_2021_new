class TrainTestData:
    def __init__(self, ratios_training_dataset,
                 ratios_validation_dataset,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 x_cong_train,
                 y_cong_train,
                 x_incong_train,
                 y_incong_train,
                 x_cong_test,
                 y_cong_test,
                 x_incong_test,
                 y_incong_test):
        self.ratios_training_dataset = ratios_training_dataset
        self.ratios_validation_dataset = ratios_validation_dataset
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_cong_train = x_cong_train
        self.y_cong_train = y_cong_train
        self.x_incong_train = x_incong_train
        self.y_incong_train = y_incong_train
        self.x_cong_test = x_cong_test
        self.y_cong_test = y_cong_test
        self.x_incong_test = x_incong_test
        self.y_incong_test = y_incong_test


class TrainResult:
    def __init__(self, curr_individual_acc,
                 curr_individual_loss,
                 curr_y_test_predictions,
                 curr_individual_model,
                 data_per_subject,
                 training_set_size,
                 validation_set_size,
                 validation_set_size_congruent):
        self.curr_individual_acc = curr_individual_acc
        self.curr_individual_loss = curr_individual_loss
        self.curr_y_test_predictions = curr_y_test_predictions
        self.curr_individual_model = curr_individual_model
        self.data_per_subject = data_per_subject
        self.training_set_size = training_set_size
        self.validation_set_size = validation_set_size
        self.validation_set_size_congruent = validation_set_size_congruent


class TrainGenomeResult:
    def __init__(self, best_individual_acc,
                 best_individual_loss,
                 individuals_models,
                 avg_accuracy,
                 data_all_subjects,
                 training_set_size,
                 validation_set_size,
                 validation_set_size_congruent):
        self.best_individual_acc = best_individual_acc
        self.best_individual_loss = best_individual_loss
        self.individuals_models = individuals_models
        self.avg_accuracy = avg_accuracy
        self.data_all_subjects = data_all_subjects
        self.training_set_size = training_set_size
        self.validation_set_size = validation_set_size
        self.validation_set_size_congruent = validation_set_size_congruent





