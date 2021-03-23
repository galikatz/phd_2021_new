from keras.models import load_model
from keras.utils.vis_utils import plot_model

def analyze_model(mode, generation, genome_id, curr_epoch, acc, loss):
	model = load_model('/Users/gali.k/phd/deepLearning/classification/DeepEvolve/model_count_gen_1_pop_1_epochs_2_acc_0.11190476233050936_loss_2.3905394576844716.h5')
	plot_model(model, to_file='model_plot_mode_{}_gens_{}_pop_{}_epochs_{}_acc_{}_loss_{}.png'.format(mode, generation, genome_id, curr_epoch, acc, loss), show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
	analyze_model('count','1', '1','2', '0.11190476233050936', '2.3905394576844716')


