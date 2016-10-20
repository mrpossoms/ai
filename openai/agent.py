import numpy as np
from collections import namedtuple
from random import randint
import random

# Obs[ 1 * 8 ] * M[8 * 4] -> Act[4, 1]
 #
 # A[ M, 1 ] * B[ N, M ] = [
 #                    [],
 # 					  [],
 # 					  [],
 # 					  []
 # 				      ]

def randcol(): return randint(0, 7)
def randrow(): return randint(0, 3)

DESC_SHAPE = (8, 4)

class Transition:
	def __init__(self,
	             pre_action_obs = None,
	             action = None,
	             reward = None,
	             done   = None,
	             post_action_obs = None):
		self.pre_action_observation = pre_action_observation
		self.post_action_observation = post_action_obs
		self.action = action
		self.reward = reward
		self.done   = done


class Agent:
	def __init__(self):
		self.last_reward = np.zeros(DESC_SHAPE)
		self.past_transitions = []
		self.params = np.random.random(DESC_SHAPE)

		self.delta_params = np.random.random(DESC_SHAPE)
		self.first = True


	def train(self, observation, reward, improved):
		self.delta_params = (np.random.random(DESC_SHAPE) * 2 - 1) * 0.1
		self.params += self.delta_params
		# self.params /= self.params.max()

	def step(self, observation, reward, done):
		observation = np.array(observation)
		action_probs = observation.dot(self.params)
		action = np.argmax(action_probs)

		this_epoch = None

		if len(self.past_transitions):
			last_transition = self.past_transitions[-1]
			last_transition.post_action_obs = observation.copy()
			last_transition.reward = reward
			last_transition.done = done

		this_epoch = Transition(
			pre_action_obs = observation.copy(),
			action = action
		)

		self.past_transitions.append(this_epoch)

		return action
