from __future__ import print_function

import gym
import numpy as np
import itertools
import os

from neat import nn, population, statistics
import operator

from hyperopt import hp, fmin, tpe, rand, space_eval

# set up environment to test on
env = gym.make('LunarLander-v2')
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'lunarLanding_config')

# search space
phenotypeSpace = [
			hp.quniform('hidden_nodes', 1, 100, 1),
			hp.uniform('max_weight', 0, 100),
			hp.uniform('min_weight', -100, 0),
			hp.quniform('feedforward', 0, 1, 1),
			hp.uniform('weight_stdev', 0, 60)
			]

average_fitnesses = []
def eval_fitness(genomes):
	fitnesses = []
	for g in genomes:
		observation = env.reset()
		env.render()
		net = nn.create_feed_forward_phenotype(g)
		observation, reward, done, info = env.step(env.action_space.sample())
		frames = 0

		fitness = 0
		
		while 1:
			inputs = observation
			output = net.serial_activate(inputs)
			index, value = max(enumerate(output), key=operator.itemgetter(1))
			action = index
			
			#print(observation)

			try:
				observation, reward, done, info = env.step(action)
			except AssertionError:
				fitness -= 1000000
				done = True
			fitness += reward
			#fitness += action
			#print(action)

			env.render()
			frames += 1

			if (frames > 500):
				done = True

			if done:
				env.reset()
				break

		# evaluate the fitness
		g.fitness = fitness
		fitnesses.append(fitness)
	average_fitnesses.append(np.mean(fitnesses))
	print(average_fitnesses)

def adjustPhenotypes(args):
	average_fitnesses[:] = []
	hidden_nodes, max_weight, min_weight, feedforward, weight_stdev = args

	pop = population.Population(config_path)
	
	pop.config.hidden_nodes = hidden_nodes
	pop.config.max_weight = max_weight
	pop.config.min_weight = min_weight
	pop.config.feedforward = feedforward
	pop.config.weight_stdev = weight_stdev

	pop = population.Population(config_path)
	pop.run(eval_fitness, 10)

	change_in_average_fitnesses = [j-i for i, j in zip(average_fitnesses[:-1], average_fitnesses[1:])]
	print(change_in_average_fitnesses)

	return np.mean(change_in_average_fitnesses)

best = fmin(adjustPhenotypes, phenotypeSpace, algo=tpe.suggest, max_evals=10)
print ("better: ", best)