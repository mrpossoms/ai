import numpy as np
from collections import namedtuple
from random import randint
import random

def randcol(): return randint(0, 7)
def randrow(): return randint(0, 3)

Transition = namedtuple('Transition', ['distance', 'observation', 'action', 'reward'])

DESC_SHAPE = (8, 4)
MAX_PAST_EPOCHS = 10000

def obs_dist(obs0, obs1):
	diff = obs0 - obs1
	return diff.dot(diff)

class Epochs:
	def __init__(self):
		self.observations = []
		self.actions = []
		self.rewards = []

	def observe_and_act(observation, action):
		self.observations += [observation]
		self.actions += [action]

	def last_action_reward(reward):
		self.rewards += [reward]

	def nearest_observations(current_observation):
		nearest_list  = []
		idx = 0

		for observation in self.observations:
			dist = obs_dist(observation, current_observation)

			if len(nearest_list) < 4:
				nearest_list += [Transition(
					distance = dist,
					observation = observation, 
					action = self.actions[idx],
					reward = self.rewards[idx],
				)]
				nearest_list = nearest_list.sorted(
					nearest_list,
					key=lambda obs: obs.distance
				)
			else:
				if dist < nearest_list[len(nearest_list) - 1].distance:
					nearest_list += [Transition(
						distance = dist,
						observation = observation, 
						action = self.actions[idx],
						reward = self.rewards[idx],
					)]
					# sort all the nearest so far
					nearest_list = nearest_list.sorted(
						nearest_list,
						key=lambda obs: obs.reward
					).reverse()

					nearest_list.pop() # keep it at 4
					nearest_list.reverse()

			idx += 1

		return nearest_list


class Agent:
	def __init__(self, exp_prob):
		self.params = np.random.random(DESC_SHAPE)
		self.delta_params = np.random.random(DESC_SHAPE)
		self.last_action_vec = np.array([ 0, 0, 0, 0 ])
		self.exp_prob = exp_prob
		self.epochs = Epochs()

	def step(self, observation, reward, done):
		observation = np.array(observation)

		action_probs = observation.dot(self.params)
	
		epochs.observe_and_act(observation, self.last_action_vec)
		epochs.last_action_reward(reward)

		nearest = epochs.nearest_observations()

		if len(nearest):
			action_probs = nearest.pop().action

			if random.random() < self.exp_prob:
				explored_actions = [ np.argmax(action) for action in nearest ]
				
				if len(explored_actions) < 4:		
					unexplored_actions = filter(lambda a: a not in explored_actions, [i for i in range(4)])
					action_probs = np.random.random((4, 1))
					action_probs[unexplored.pop()][0] += 1	

		
		#action_probs = observation.dot(self.params)
		self.last_action_vec = action_probs
		action = np.argmax(action_probs)

		return action
