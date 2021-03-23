import matlab.engine
import argparse


def main(arguments):
	eng = matlab.engine.start_matlab()
	eng.addpath('/Users/gali.k/phd/Genereating_dot_arrays')
	eng.pipeline_from_python(arguments.congruency, arguments.equate, arguments.savedir, str(arguments.index), nargout=0)
	# eng.pipeline(nargout=0)
	eng.quit()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='evolve arguments')
	parser.add_argument('--congruency', dest='congruency', type=int, required=True,
						help='The stimuli is congruent or incongruent')
	parser.add_argument('--equate', dest='equate', type=int, required=True,
						help='1 is for average diameter; 2 is for total surface area; 3 is for convex hull')
	parser.add_argument('--savedir', dest='savedir', type=str, required=True, help='The save dir')
	parser.add_argument('--index', dest='index', type=str, required=True, help='The save dir index = generation index')
	args = parser.parse_args()
	main(args)

