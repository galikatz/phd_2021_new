from sklearn.metrics import roc_auc_score
from train import creating_train_test_data
from keras.models import load_model
import glob
import os


def load_keras_model_from_h5_file(hd5_path):
    loaded_models = {}
    model_files = glob.glob(hd5_path + os.sep + f"*.h5")
    for model_file in model_files:
        loaded_model = load_model(model_file, compile=False)
        loaded_models.update({model_file: loaded_model})
    return loaded_models


def load_models_for_genomes(genomes, hd5_path):
    loaded_models = {}
    genome_id_to_genome = convert_to_dict(genomes)
    model_files = glob.glob(hd5_path + os.sep + f"*.h5")
    for model_file in model_files:
        loaded_model = load_model(model_file, compile=False)
        genome_id = int(extract_genome_id_from_file(model_file))
        if genome_id in genome_id_to_genome.keys():
            loaded_models.update({genome_id_to_genome[genome_id]: loaded_model})
    return loaded_models


def extract_genome_id_from_file(model_file_name):
    individual_id = model_file_name[model_file_name.find('individual_') + 11:model_file_name.find('_acc')]
    return individual_id


def convert_to_dict(list_of_genomes):
    dict_of_genome_id_to_genome = {}
    for genome in list_of_genomes:
        dict_of_genome_id_to_genome.update({genome.u_ID: genome})
    return dict_of_genome_id_to_genome


def get_physical_properties_to_load(images_dir, equate):
    if equate == 1:
        return [StimuliData(images_dir.replace("equate_1", "equate_2") + "_1", "equate_2"), StimuliData(images_dir.replace("equate_1", "equate_3") + "_1", "equate_3")]
    elif equate == 2:
        return [StimuliData(images_dir.replace("equate_2", "equate_1") + "_1", "equate_1"), StimuliData(images_dir.replace("equate_2", "equate_3") + "_1", "equate_3")]
    return [StimuliData(images_dir.replace("equate_3", "equate_2") + "_1", "equate_2"), StimuliData(images_dir.replace("equate_3", "equate_1") + "_1", "equate_1")]


class StimuliData:
    def __init__(self, path, equate):
        self.path = path
        self.equate = equate


if __name__ == '__main__':
    hd5_path = "/Users/gali.k/phd/phd_2021/results/equate_1/size/best_model_2022-05-19_14_mode_size_equate_1_gen_5_individual_90_acc_0.983_loss_0.435_layers_3_neurons_[128, 32, 32]_activation_linear_optimizer_nadam.h5"
    loaded_model = load_keras_model_from_h5_file(hd5_path)
    list_of_stimuli_data = get_physical_properties_to_load("size", "equate_1")

    mode = "size"
    batch_size = 500

    for stumli_data in list_of_stimuli_data:
        (x_train, y_train), (x_test, y_test), (x_cong_train, y_cong_train), (x_incong_train, y_incong_train), (
        x_cong_test, y_cong_test), (
        x_incong_test, y_incong_test), ratios_training_dataset, ratios_validation_dataset = creating_train_test_data(
            dir=stumli_data.path, stimuli_type="katzin", mode=stumli_data.mode, nb_classes=2)
        y_test_prediction = loaded_model.predict(x=x_test, batch_size=batch_size, verbose=0)
        auc = roc_auc_score(y_test_prediction, x_test)
        print(f"Roc score: {auc}")