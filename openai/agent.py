import numpy as np
from collections import namedtuple
from random import randint
import random

TRAINING_EPOCHS = 200
GENERATION_SIZE = 500
RANDOMIZATION = 2


DESC_SHAPE = (8, 4)

def rnd_white(base=None, mag=1):
	matrix = (np.random.random(DESC_SHAPE) * 2 - 1) * mag

	if base is not None:
		matrix = matrix + base

	return matrix

def rnd_gaussian(base=None, mag=1, std_dev=0.25):
	matrix = (np.random.normal(0, std_dev, DESC_SHAPE) * 2 - 1) * mag

	if base is not None:
		matrix = matrix + base

	return matrix

class Agent:
	def __init__(self, env, maximize=True):
		self.environment = env
		self.best_candidate = None
		self.candidates = []
		self.maximize = maximize
		self.gen_candidates(std_dev=1)
		self.improvement_rate = 0

		try:
			self.best_candidate = best_candidate = np.loadtxt(fname='params.txt', delimiter = ",")
			self.candidates += [best_candidate]
			print("Loaded existing params")
		except:
			pass

	def gen_candidates(self, base=None, std_dev=0.25):
		self.candidates = []

		if isinstance(base, np.ndarray):
			self.candidates += [base]
			for _ in range(GENERATION_SIZE):
				self.candidates += [rnd_gaussian(base=base, mag=RANDOMIZATION, std_dev=std_dev)]
		elif isinstance(base, list) and isinstance(base[0], np.ndarray):
			self.candidates += base

			for i in range(GENERATION_SIZE):
				self.candidates += [rnd_gaussian(base=base[i % len(base)], mag=RANDOMIZATION, std_dev=std_dev)]
		else:
			for _ in range(GENERATION_SIZE):
				self.candidates += [rnd_gaussian(mag=RANDOMIZATION, std_dev=std_dev)]

	def run_epoch(self, desc_matrix=None, show=False):
		total_reward = 0

		if desc_matrix is None:
			desc_matrix = self.best_candidate

		self.environment.reset()
		observation, reward, done, info = self.environment.step(0)			

		for _ in range(600):
			observation = np.array(observation)
			action_probs = observation.dot(desc_matrix)
			
			if show:
				self.environment.render()
			observation, reward, done, info = self.environment.step(np.argmax(action_probs))
			total_reward += reward

			if done:
				break

		return total_reward

	def evaluate_generation(self):
		candidate_score = []
		avg_score = 0

		# score the existing candidates
		for candidate in self.candidates:
			score = self.run_epoch(desc_matrix=candidate)
			avg_score += score / float(len(self.candidates))

			candidate_score += [(candidate, score)]

		candidate_score = sorted(candidate_score, key=lambda cs_tuple: cs_tuple[1])

		best_candidates = []
		for _ in range(5):
			candidate = candidate_score.pop()
			best_candidates += [(candidate[0].copy(), candidate[1])]

		return best_candidates, avg_score

	def solve(self):
		last_gen_avg = 0

		for i in range(TRAINING_EPOCHS):
			best_candidates, avg_score = self.evaluate_generation()

			if i is 0:
				last_gen_avg = avg_score

			self.improvement_rate = (avg_score - last_gen_avg)
			last_gen_avg = avg_score

			best_best = best_candidates[len(best_candidates) - 1]
			print("(%d/%d) Best score: %f Avg: %f Improvement: %f" % (i, TRAINING_EPOCHS, best_best[1], avg_score, self.improvement_rate))

			if avg_score >= 200:
				break
			best_matrices = []
			for candidate in best_candidates: best_matrices += [candidate[0]]

			self.gen_candidates(base=best_matrices, std_dev = 10. / max(self.improvement_rate, 1))
			self.best_candidate = best_matrices.pop()
	
			np.savetxt(fname='params.txt', X=self.best_candidate, delimiter=',')
