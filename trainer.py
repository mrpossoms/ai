import numpy as np
from threading import Thread

class Subject:
	def __init__(self, rep):
		self.soln = rep
		self.g_soln_space = np.zero(rep.shape)
		self.d_score = np.zero(rep.shape) 


class Trainer:
	def __init__(self, maximizing=True):
                self.maximizing = maximizing
		self.gamma = 0.001 # learning rate
		self.thread_count = 4

	def compute_gradient(self, candidate, col_range, resource):
		score_0 = self.avg_evaluation(candidate.soln, resource)

		for x in col_range: 
			for y in range(candidate.shape[1]):
				dxy = candidate.soln.copy()
				dxy[x][y] += self.gamma
				diff = score_0 - self.avg_evaluation(dxy, resource)  
				candidate.d_score = diff
				candidate.g_soln_space[x][y] = (self.gamma * self.diff) / self.gamma

        def subprocess_resource(self):
            raise NotImplemented

	def train(self, iterations, candidate):
		col_per_thread = candidate.soln.shape[0]
                threads = []

		for i in range(self.thread_count):
			thread = Thread(target=self.compute_gradient, kwargs={
                                "self": self,
                                "candidate": candidate,
                                "col_range": range(i * col_per_thread, (i + 1) * col_per_thread),
                                "resource": self.subprocess_resource()
                            })

                        threads += [thread]

                for i in range(iterations):
                    # start computing gradients
                    for thread in threads:
                        thread.run()

                    for thread in threads:
                        thread.join()
                       
                    # modify solution matrix values by gradient
                    # computed in the previous run
                    candidate.soln += candidate.g_soln_space


	def evaluate(self, candiate):
		raise NotImplemented

	def avg_evaluation(self, candiate, iterations=10):
		score = 0
		for _ in range(iterations):
			score += self.evaluate(candidate)
		
		return score / iterations


	
