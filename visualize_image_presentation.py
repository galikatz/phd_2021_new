from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from matplotlib._image import LANCZOS

import models
from classify import crop_center, IMG_SIZE
from PIL import Image
from keras.models import Model
from keras.preprocessing.image import image_utils

from keras.models import load_model


def process_image(path):
    rgba_image = Image.open(path)
    rgb_image = rgba_image.convert('RGB')
    rgb_image = crop_center(rgb_image)
    rgb_image = rgb_image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    return rgb_image


def predict_one_input(model, img_path):
    img = image_utils.img_to_array(process_image(img_path))
    x = np.expand_dims(img, axis=0)
    y_pred = model.predict(x)
    predicted_class = np.argmax(y_pred, axis=1)
    return predicted_class


def visualize_image_compact_rep(model, img_path):
    # Convert the image to a numpy array
    img = image_utils.img_to_array(process_image(img_path))

    x = np.expand_dims(img, axis=0)
    num_of_layers = len(model.layers)
    # Get the activations for the selected intermediate layer
    layer_outputs = [layer.output for layer in model.layers[:num_of_layers]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x)

    layer_names = []
    for layer in model.layers[:num_of_layers]:
        layer_names.append(layer.name)

    print(layer_names)
    for layer_name, layer_activation in zip(layer_names, activations):
        # for i, activation in enumerate(activations):
        plt.figure()
        plt.title(f"{layer_name}")
        print(layer_name)
        # plt.imshow(layer_activation)
        plt.imshow(layer_activation[0, :, :, 0])
        plt.grid(None)
        plt.show()


def visualize_image_rep(model, img_path):

    # Convert the image to a numpy array
    img = image_utils.img_to_array(process_image(img_path))

    x = np.expand_dims(img, axis=0)

    # Get the activations for the selected intermediate layer
    layer_outputs = [layer.output for layer in model.layers[:-1]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x)


    # Visualize the activations
    layer_names = []
    for layer in model.layers[:-1]:
        layer_names.append(layer.name)

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

if __name__ == '__main__':

    # loaded_model = load_model("/Users/gali.k/phd/phd_2021/visulization/simulation_5_balanced/equate_2/count-size/gens/model_2023-02-04_15_mode_size_equate_2_gen_30_individual_582_acc_0.5_loss_41.461_layers_4_neurons_[128, 32, 128, 128]_activation_linear_optimizer_rmsprop.h5", compile=True)

    # loaded_model = load_model("/Users/gali.k/phd/phd_2021/visulization/simulation_5_balanced/equate_2/count-size/gens/model_2023-02-04_15_mode_size_equate_2_gen_30_individual_584_acc_0.946_loss_0.576_layers_2_neurons_[128, 32]_activation_tanh_optimizer_nadam.h5", compile=True)
    #
    # # loaded_model = load_model("/Users/gali.k/phd/phd_2021/visulization/simulation_5_balanced/equate_2/size/gens/model_2023-02-03_20_mode_size_equate_2_gen_15_individual_295_acc_0.908_loss_0.475_layers_3_neurons_[32, 16, 64]_activation_elu_optimizer_adam.h5")
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_1/images_1/incong56_5_9_999.jpg"
    # # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_1/images_1/incong71_7_5_40.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # visualize_image_rep(loaded_model, img_path)
    # print(y_pred_class)

    #AD
    # loaded_model = load_model("/Users/gali.k/phd/phd_2021/models/model_2023-02-11_10_mode_count_equate_1_gen_1_individual_4_acc_0.496_loss_1.751_layers_5_neurons_[32, 128, 64, 128, 128]_activation_hard_sigmoid_optimizer_adam.h5", compile=False)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_1/images_19/incong50_10_5_1043.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # visualize_image_rep(loaded_model, img_path)
    # print(y_pred_class)

    #Train with AD but test with CH
    # loaded_model = load_model("/Users/gali.k/phd/phd_2021/models/model_2023-02-11_10_mode_count_equate_1_gen_1_individual_4_acc_0.496_loss_1.751_layers_5_neurons_[32, 128, 64, 128, 128]_activation_hard_sigmoid_optimizer_adam.h5", compile=True)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_3/images_2/incong63_8_5_144.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # # visualize_image_compact_rep(loaded_model, img_path)
    # print(y_pred_class)

    # #CH
    # loaded_model = load_model(
    #     "/Users/gali.k/phd/phd_2021/models/model_2023-02-11_08_mode_count_equate_1_gen_1_individual_4_acc_0.5_loss_28.221_layers_5_neurons_[128, 32, 128, 128, 128]_activation_tanh_optimizer_sgd.h5",
    #     compile=False)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_3/images_1/incong50_5_10_1229.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # visualize_image_rep(loaded_model, img_path)
    # print(y_pred_class)

    # CH individual with accuracy 74% that was tested with AD
    # we will show a correct classification of 71 ratio:
    # loaded_model = load_model(
    # "/Users/gali.k/phd/phd_2021/models/model_2023-02-11_06_mode_count_equate_3_gen_1_individual_4_acc_0.738_loss_0.598_layers_2_neurons_[16, 64]_activation_linear_optimizer_adamax.h5",
    # compile=False)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_1/images_1/incong71_7_5_737.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # visualize_image_rep(loaded_model, img_path)
    # print(y_pred_class)

    # TS individual with accuracy 100% that was tested with AD
    # loaded_model = load_model(
    # "/Users/gali.k/phd/phd_2021/models/model_2023-02-11_14_mode_colors_equate_2_gen_1_individual_1_acc_1.0_loss_0.059_layers_2_neurons_[64, 32]_activation_tanh_optimizer_rmsprop.h5",
    # compile=False)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_1/images_1/incong71_7_5_737.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # visualize_image_rep(loaded_model, img_path)
    # print(y_pred_class)

    # # AD individual with accuracy 87.9% that was tested with CH  - we can see that physical represenation is larget for the smaller number!
    # loaded_model = load_model(
    # "/Users/gali.k/phd/phd_2021/models/model_2023-02-11_07_mode_count_equate_1_gen_1_individual_1_acc_0.879_loss_0.41_layers_3_neurons_[64, 64, 128]_activation_elu_optimizer_nadam.h5",
    # compile=False)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_3/images_10/incong63_8_5_145.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # visualize_image_rep(loaded_model, img_path)
    # print(y_pred_class)


    # PNAS
    # BEST
    # num-CNN processing incong 50 ratio
    # loaded_model = load_model("/Users/gali.k/phd/phd_2021/models/PNAS/count/model_2023-02-03_11_mode_count_equate_2_gen_30_individual_596_acc_0.85_loss_0.415_layers_5_neurons_[32, 16, 64, 128, 64]_activation_tanh_optimizer_adam.h5",
    #                           compile=False)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_2/images_4/incong50_10_5_1796_.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # # visualize_image_rep(loaded_model, img_path)
    # visualize_image_compact_rep(loaded_model, img_path)
    # print(y_pred_class)

    # pys-num-CNN processing incong 50 ratio
    # loaded_model = load_model("/Users/gali.k/phd/phd_2021/models/PNAS/size-count/model_2023-02-05_13_mode_count_equate_2_gen_30_individual_592_acc_0.921_loss_0.283_layers_5_neurons_[64, 64, 128, 32, 64]_activation_linear_optimizer_adamax.h5",
    #                           compile=False)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_2/images_4/incong50_10_5_1796_.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # # visualize_image_rep(loaded_model, img_path)
    # visualize_image_compact_rep(loaded_model, img_path)
    # print(y_pred_class)

    # WORST
    # num-CNN processing incong 50 ratio
    loaded_model = load_model("/Users/gali.k/phd/phd_2021/models/PNAS/worst/count/model_2023-02-03_11_mode_count_equate_2_gen_30_individual_589_acc_0.504_loss_2.516_layers_2_neurons_[32, 64]_activation_tanh_optimizer_rmsprop.h5",
                              compile=False)
    img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_2/images_4/incong50_10_5_1796_.jpg"
    y_pred_class = predict_one_input(loaded_model, img_path)
    # visualize_image_rep(loaded_model, img_path)
    # visualize_image_compact_rep(loaded_model, img_path)
    print(y_pred_class)



    #  pys-num-CNN processing incong 50 ratio
    # loaded_model = load_model("/Users/gali.k/phd/phd_2021/models/PNAS/worst/size-count/model_2023-02-05_13_mode_count_equate_2_gen_30_individual_581_acc_0.5_loss_0.827_layers_5_neurons_[64, 128, 128, 128, 32]_activation_sigmoid_optimizer_nadam.h5",
    #                           compile=False)
    # img_path = "/Users/gali.k/phd/phd_2021/stimuli/equate_2/images_4/incong50_10_5_1796_.jpg"
    # y_pred_class = predict_one_input(loaded_model, img_path)
    # # visualize_image_rep(loaded_model, img_path)
    # # visualize_image_compact_rep(loaded_model, img_path)
    # print(y_pred_class)
