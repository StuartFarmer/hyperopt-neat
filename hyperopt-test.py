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
			hp.uniform('weight_stdev', 0, 60),
			
			hp.quniform('pop_size', 20, 100, 1),
			
			hp.uniform('prob_add_conn',0, 1),
			hp.uniform('prob_add_node',0, 1),
			hp.uniform('prob_delete_conn',0, 1),
			hp.uniform('prob_delete_node',0, 1),
			hp.uniform('prob_mutate_bias',0, 1),
			hp.uniform('bias_mutation_power',0, 1),
			hp.uniform('prob_mutate_response',0, 1),
			hp.uniform('response_mutation_power',0, 1),
			hp.uniform('prob_mutate_weight',0, 1),
			hp.uniform('prob_replace_weight',0, 1),
			hp.uniform('weight_mutation_power',0, 1),
			hp.uniform('prob_mutate_activation',0, 1),
			hp.uniform('prob_toggle_link',0, 1),

			hp.uniform('compatibility_threshold', 0, 100),
			hp.uniform('excess_coefficient', 0, 100),
			hp.uniform('disjoint_coefficient', 0, 100),
			hp.uniform('weight_coefficient', 0, 100),

			hp.quniform('max_stagnation', 0, 100, 1),

			hp.quniform('elitism', 0, 100, 1),
			hp.uniform('survival_threshold', 0, 1)

			hp.uniform('pos_x', -100, 100)
			hp.uniform('pos_y', -100, 100)
			hp.uniform('vel_x', -100, 100)
			hp.uniform('vel_y', -100, 100)
			hp.uniform('angle', -100, 100)
			hp.uniform('angle_vel', -100, 100)
			hp.uniform('leg_one', -100, 100)
			hp.uniform('leg_two', -100, 100)
			hp.uniform('total_frames', -100, 100)
			hp.uniform('fuel_use', -100, 100)
			hp.uniform('reward', -100, 100)
			]

fitness_weights = []
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
	fitness_weights[:] = []
	average_fitnesses[:] = []
	hidden_nodes, max_weight, min_weight, feedforward, weight_stdev, pop_size, prob_add_conn, prob_add_node, prob_delete_conn, prob_delete_node, prob_mutate_bias, bias_mutation_power, prob_mutate_response, response_mutation_power, prob_mutate_weight, prob_replace_weight, weight_mutation_power, prob_mutate_activation, prob_toggle_link, compatibility_threshold, excess_coefficient, disjoint_coefficient, weight_coefficient, max_stagnation, elitism, survival_threshold = args

	pop = population.Population(config_path)
	
	pop.config.hidden_nodes = hidden_nodes
	pop.config.max_weight = max_weight
	pop.config.min_weight = min_weight
	pop.config.feedforward = feedforward
	pop.config.weight_stdev = weight_stdev

	pop.config.pop_size = pop_size
	pop.config.prob_add_conn = prob_add_conn
	pop.config.prob_add_node = prob_add_node
	pop.config.prob_delete_conn = prob_delete_conn
	pop.config.prob_delete_node = prob_delete_node
	pop.config.prob_mutate_bias = prob_mutate_bias
	pop.config.bias_mutation_power = bias_mutation_power
	pop.config.prob_mutate_response = prob_mutate_response
	pop.config.response_mutation_power = response_mutation_power
	pop.config.prob_mutate_weight = prob_mutate_weight
	pop.config.prob_replace_weight = prob_replace_weight
	pop.config.weight_mutation_power = weight_mutation_power
	pop.config.prob_mutate_activation = prob_mutate_activation
	pop.config.prob_toggle_link = prob_toggle_link
	pop.config.compatibility_threshold = compatibility_threshold
	pop.config.excess_coefficient = excess_coefficient
	pop.config.disjoint_coefficient = disjoint_coefficient
	pop.config.weight_coefficient = weight_coefficient
	pop.config.max_stagnation = max_stagnation
	pop.config.elitism = elitism
	pop.config.survival_threshold = survival_threshold

	#pop = population.Population(config_path)
	pop.run(eval_fitness, 10)

	change_in_average_fitnesses = [j-i for i, j in zip(average_fitnesses[:-1], average_fitnesses[1:])]
	print(change_in_average_fitnesses)

	return np.mean(change_in_average_fitnesses)

best = fmin(adjustPhenotypes, phenotypeSpace, algo=tpe.suggest, max_evals=10)
print ("better: ", best)