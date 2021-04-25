import matlab.engine
import argparse
import fnmatch
import os

def main(arguments):
	eng = matlab.engine.start_matlab()
	eng.addpath('/Users/gali.k/phd/Genereating_dot_arrays')
	eng.pipeline_from_python(arguments.congruency, arguments.equate, arguments.savedir, str(arguments.index), nargout=0)
	# eng.pipeline(nargout=0)
	eng.quit()


def generate_new_images(congruency, equate, savedir, index, prefix=None):
	eng = matlab.engine.start_matlab()
	eng.addpath('/Users/gali.k/phd/Genereating_dot_arrays')
	eng.pipeline_from_python(congruency, equate, savedir, str(index), nargout=0)
	eng.quit()
	if prefix:
		number_files_created = len(fnmatch.filter(os.listdir(savedir + "_" + str(index)), prefix +'*.jpg'))
	else:
		number_files_created = len(fnmatch.filter(os.listdir(savedir + "_" + str(index)), '*.jpg'))
	return number_files_created



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='evolve arguments')
	parser.add_argument('--congruency', dest='congruency', type=int, required=True,
						help='The stimuli is congruent or incongruent, 0-incong, 1-cong')
	parser.add_argument('--equate', dest='equate', type=int, required=True,
						help='1 is for average diameter; 2 is for total surface area; 3 is for convex hull')
	parser.add_argument('--savedir', dest='savedir', type=str, required=True, help='The save dir')
	parser.add_argument('--index', dest='index', type=str, required=True, help='The save dir index = generation index')
	args = parser.parse_args()
	main(args)
