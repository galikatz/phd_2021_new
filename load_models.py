from keras.models import load_model
from keras.utils.vis_utils import plot_model

def analyze_model(mode, generation, genome_id, curr_epoch, acc, loss):
	model = load_model('/Users/gali.k/phd/phd_2021/results/equate_1/size/best_model_2022-05-19_14_mode_size_equate_1_gen_5_individual_90_acc_0.983_loss_0.435_layers_3_neurons_[128, 32, 32]_activation_linear_optimizer_nadam.h5')
	plot_model(model, to_file='model_plot_mode_{}_gens_{}_pop_{}_epochs_{}_acc_{}_loss_{}.png'.format(mode, generation, genome_id, curr_epoch, acc, loss), show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
	analyze_model('count','1', '1','2', '0.11190476233050936', '2.3905394576844716')


