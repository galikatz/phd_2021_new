
import argparse
import fnmatch
import os
import logging


def main(arguments):
	# This is here because on cloud mode we do not want to import this
	import matlab.engine
	eng = matlab.engine.start_matlab()
	eng.addpath('/Users/gali.k/phd/Genereating_dot_arrays')
	eng.pipeline_from_python(arguments.congruency, arguments.equate, arguments.savedir, str(arguments.index), nargout=0)
	# eng.pipeline(nargout=0)
	eng.quit()


def generate_new_images(congruency, equate, savedir, index, prefix=None, ratio=50):
	# This is here because on cloud mode we do not want to import this
	import matlab.engine
	eng = matlab.engine.start_matlab()
	eng.addpath('/Users/gali.k/phd/Genereating_dot_arrays')
	generating_stimuli = True
	retries = 0
	while generating_stimuli and retries < 5:
		try:
			eng.pipeline_from_python(congruency, equate, savedir, str(index), ratio, nargout=0)
			generating_stimuli = False
		except Exception as e:
			logging.error("Timeout exception trying again for index: %s and ratio: %s" % (index, ratio))
			retries += 1
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
