"""
Class that holds a genetic algorithm for evolving a network.

Inspiration:

	http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from __future__ import print_function

import random
import logging
import copy

from functools  import reduce
from operator   import add
from genome     import Genome
from idgen      import IDgen
from allgenomes import AllGenomes


class Evolver():
	"""Class that implements genetic algorithm."""

	def __init__(self, all_possible_genes, retain=0.9, random_select=0.1, mutate_chance=0.25):
		"""Create an optimizer.

		Args:
			all_possible_genes (dict): Possible genome parameters
			retain (float): Percentage of population to recombine with each other (the top best)
			random_select (float): Probability of a rejected genome
				remaining in the population
			mutate_chance (float): Probability a genome will be
				randomly mutated

		"""

		self.all_possible_genes = all_possible_genes
		self.retain             = retain
		self.random_select      = random_select
		self.mutate_chance      = mutate_chance

		#set the ID gen
		self.ids = IDgen()

	def create_population(self, count):
		"""Create a population of random networks.

		Args:
			count (int): Number of networks to generate, aka the
				size of the population

		Returns:
			(list): Population of network objects

		"""
		pop = []

		i = 0

		while i < count:

			# Initialize a new genome.
			genome = Genome( self.all_possible_genes, {}, self.ids.get_next_ID(), 0, 0, self.ids.get_Gen() )

			# Set it to random parameters.
			genome.set_genes_random()

			if i == 0:
				#this is where we will store all genomes
				self.master = AllGenomes( genome )
			else:
				# Make sure it is unique....
				while self.master.is_duplicate( genome ):
					genome.mutate_one_gene()

			# Add the genome to our population.
			pop.append(genome)

			# and add to the master list
			if i > 0:
				self.master.add_genome(genome)

			i += 1

		#self.master.print_all_genomes()

		#exit()

		return pop

	@staticmethod
	def fitness(genome):
		"""Return the accuracy, which is our fitness function."""
		return genome.accuracy

	def grade(self, pop):
		"""Find average fitness for a population.

		Args:
			pop (list): The population of networks/genome

		Returns:
			(float): The average accuracy of the population

		"""
		summed = reduce(add, (self.fitness(genome) for genome in pop))
		return summed / float((len(pop)))

	def breed(self, mom, dad):
		"""Make two children from parental genes.

		Args:
			mother (dict): genome parameters
			father (dict): genome parameters

		Returns:
			(list): Two network objects

		"""
		children = []

		#where do we recombine? 0, 1, 2, 3, 4... N?
		#with four genes, there are three choices for the recombination
		# ___ * ___ * ___ * ___
		#0 -> no recombination, and N == length of dictionary -> no recombination
		#0 and 4 just (re)create more copies of the parents
		#so the range is always 1 to len(all_possible_genes) - 1
		pcl = len(self.all_possible_genes)
		# the recomb_loc is the index where we decide to recombine and switch between mom and dad's genes
		recomb_loc = random.randint(1,pcl - 1)

		#for _ in range(2): #make _two_ children - could also make more
		child1 = {}
		child2 = {}

		#enforce defined genome order using list
		#keys = ['nb_neurons', 'nb_layers', 'activation', 'optimizer']
		keys = list(self.all_possible_genes)
		keys = sorted(keys) #paranoia - just to make sure we do not add unintentional randomization

		#*** CORE RECOMBINATION CODE ****
		for x in range(0, pcl):
			if x < recomb_loc:
				child1[keys[x]] = mom.geneparam[keys[x]]
				child2[keys[x]] = dad.geneparam[keys[x]]
			else:
				child1[keys[x]] = dad.geneparam[keys[x]]
				child2[keys[x]] = mom.geneparam[keys[x]]

		# Initialize a new genome
		# Set its parameters to those just determined
		# they both have the same mom and dad
		genome1 = Genome( self.all_possible_genes, child1, self.ids.get_next_ID(), mom.u_ID, dad.u_ID, self.ids.get_Gen() )
		genome2 = Genome( self.all_possible_genes, child2, self.ids.get_next_ID(), mom.u_ID, dad.u_ID, self.ids.get_Gen() )

		#at this point, there is zero guarantee that the genome is actually unique

		# Randomly mutate one gene
		if self.mutate_chance > random.random():
			logging.info("##### Mutating one gene of genome: %s ####" % genome1.u_ID)
			genome1.mutate_one_gene()

		if self.mutate_chance > random.random():
			logging.info("##### Mutating one gene of genome: %s ####" % genome2.u_ID)
			genome2.mutate_one_gene()

		#do we have a unique child or are we just retraining one we already have anyway?
		while self.master.is_duplicate(genome1):
			genome1.mutate_one_gene()

		self.master.add_genome(genome1)

		while self.master.is_duplicate(genome2):
			genome2.mutate_one_gene()

		self.master.add_genome(genome2)

		children.append(genome1)
		children.append(genome2)

		return children

	def evolve(self, pop):
		"""Evolve a population of genomes.

		Args:
			pop (list): A list of genome parameters

		Returns:
			(list): The evolved population of networks

		"""
		#increase generation
		self.ids.increase_Gen()

		# Get scores for each genome
		graded = [(self.fitness(genome), genome) for genome in pop]

		#and use those scores to fill in the master list
		for genome in pop:
			self.master.set_accuracy(genome)

		# Sort on the scores.
		graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

		# Get the number we want to reproduce for next cycle
		retain_length = int(len(graded)*self.retain)

		# in real life we do not retain individuals from one gen to another,
		# Thus the retain is 0 and we are doing 100% recommbination and 25% mutation.
		if retain_length < 2:
			retain_length = 2 # we need at least 2 parents to recombine for creating the next generation

		# the best parents were chosen
		selection_pool = graded[:retain_length]
		new_generation = []

		# We keep the same population size every generation

		desired_length = len(pop)

		children       = []

		# Add children, which are bred from pairs of remaining (i.e. very high or lower scoring) genomes.
		while len(children) < desired_length:
			# Get a random mom and dad, but, need to make sure they are distinct
			parents  = random.sample(range(len(selection_pool)), k=2)

			i_male   = parents[0]
			i_female = parents[1]

			male   = selection_pool[i_male]
			female = selection_pool[i_female]

			# Recombine and mutate
			babies = self.breed(male, female)
			# the babies are guaranteed to be novel
			# add only the number of children that is needed to keep the same population size
			for i in range (0,2): #only 2 babies at a time
				if len(children) < desired_length:
					children.append(babies[i])


		new_generation.extend(children)

		return new_generation
