from sklearn.metrics import roc_auc_score
from train import creating_train_test_data
from keras.models import load_model
import glob
import os

PREF = "/Users/gali.k/phd/phd_2021/stimui/"
PREF_COLORS = "/Users/gali.k/phd/phd_2021/stimui/colors/"


def load_keras_model_from_h5_file(hd5_path):
    loaded_models = {}
    model_files = glob.glob(hd5_path + os.sep + f"*.h5")
    for model_file in model_files:
        loaded_model = load_model(model_file, compile=False)
        loaded_models.update({model_file: loaded_model})
    return loaded_models


def get_physical_propeties_to_load(is_colors, hd5_path):
    if is_colors is True:
        prefix = PREF_COLORS
    else:
        prefix = PREF
    if "equate_1" in hd5_path:
        return [prefix + "equate_2", prefix + "equate_3"]
    elif "equate_2" in hd5_path:
        return [prefix + "equate_1", prefix + "equate_3"]
    return [prefix + "equate_2", prefix + "equate_3"]


if __name__ == '__main__':
    hd5_path = "/Users/gali.k/phd/phd_2021/results/equate_1/size/best_model_2022-05-19_14_mode_size_equate_1_gen_5_individual_90_acc_0.983_loss_0.435_layers_3_neurons_[128, 32, 32]_activation_linear_optimizer_nadam.h5"
    loaded_model = load_keras_model_from_h5_file(hd5_path)
    stumli_paths = get_physical_propeties_to_load("regular", hd5_path)

    mode = "size"
    batch_size = 500

    for stumli_path in stumli_paths:
        (x_train, y_train), (x_test, y_test), (x_cong_train, y_cong_train), (x_incong_train, y_incong_train), (
        x_cong_test, y_cong_test), (
        x_incong_test, y_incong_test), ratios_training_dataset, ratios_validation_dataset = creating_train_test_data(
            dir=stumli_path, stimuli_type="katzin", mode=mode, nb_classes=2)
        y_test_prediction = loaded_model.predict(x=x_test, batch_size=batch_size, verbose=0)
        auc = roc_auc_score(y_test_prediction, x_test)
        print(f"Roc score: {auc}")