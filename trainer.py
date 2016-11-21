import numpy as np
from Threading import Thread

class Subject:
	def __init__(self, rep, maximizing=True):
		self.soln = rep
		self.g_soln_space = np.zero(rep.shape)
		self.d_score = np.zero(rep.shape) 
		self.gamma = 0.001 # learning rate
	
		self.thread_count = 4

	def compute_gradient(self, col_range=range(self.soln.shape[0])):
		cand = self.soln.copy()
		score_0 = self.avg_evaluation(cand)

		for x in col_range: 
			for y in range(cand.shape[1]):
				dxy = cand.copy()
				dxy[x][y] += self.gamma
				diff = score_0 - self.avg_evaluation(dxy)  
				self.d_score = diff
				self.g_soln_space[x][y] = (self.gamma * self.diff) / self.gamma

	def train(self):
		col_per_thread = self.soln.shape[0]

		for i in range(self.thread_count):
			t = Thread()



	def evaluate(self, candiate):
		raise NotImplemented

	def avg_evaluation(self, candiate, iterations=10):
		score = 0
		for _ in range(iterations):
			score += self.evaluate(candidate)
		
		return score / iterations

class Trainer:
	def __init__(self):
		pass

	
