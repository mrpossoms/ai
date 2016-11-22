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
                self.os = ObservationSpace(0.25)

	def compute_gradient(self, candidate, col_range, resource):
		for x in col_range: 
			for y in range(candidate.shape[1]):
				dxy = candidate.soln.copy()
				dxy[x][y] += self.gamma
				diff = self.diff(candidate.soln, dxy, resource)  
				candidate.d_score = diff
				candidate.g_soln_space[x][y] = (self.gamma * self.diff) / self.gamma

        def subprocess_resource(self, subproc_idx):
            raise NotImplemented

	def train(self, iterations, candidate):
		col_per_thread = candidate.soln.shape[0]
                threads = []

                # initialize threads, providing them with the needed resources
                # and values to perform gradient computations
		for i in range(self.thread_count):
			thread = Thread(target=self.compute_gradient, kwargs={
                                "self": self,
                                "candidate": candidate,
                                "col_range": range(i * col_per_thread, (i + 1) * col_per_thread),
                                "resource": self.subprocess_resource()
                            })

                        threads += [thread]

                for i in range(iterations):
                    # reset the observation-space table
                    self.os.clear()
                    
                    # start computing gradients
                    for thread in threads:
                        thread.run()

                    # finish computing gradients
                    for thread in threads:
                        thread.join()
                       
                    # modify solution matrix values by gradient
                    # computed in the previous run
                    candidate.soln += candidate.g_soln_space

        def diff(self, old_candidate, new_candidate, resource):
            while True:
                # generate some trial results with the original candidate
                # store them in the observation-space table
                for _ in range(10):
                    result = self.evaluate(old_candidate, resource) 
                    self.os += result

                # run the altered candidate, collect the starting conditions and score
                start, score = self.evaluate(new_candidate, resource)
                nearest = self.os[start]

                # do we have any nearby inital observations from the old
                # candidate trails?
                if len(nearest) > 0:
                    old_start, old_score, distance = nearest[0]
                    return score - old_score

        # Note: evaluate must return a tuple containing the score
        #       and the initial state of the observation vector
	def evaluate(self, candiate, resource):
		raise NotImplemented

class ObservationSpace():
    def __init__(self, match_threshold):
        self.observations = []
        self.threshold = match_threshold
        self.lock = threading.Lock()

    def clear(self):
        self.observations = []

    def __add__(self, other):
        self.lock.acquire()
        self.observations += other
        self.lock.release()

    def __getitem_(self, key):
        nearest = []

        self.lock.acquire()
        for (observation, score) in self.observations:
            dist = np.linalg.norm(key - observation)
            if dist < self.threshold:
                nearest += [(observation, score, dist)]
        self.lock.release()

        # ordered from nearest to most distant
        return sorted(nearest, key=lambda n_tuple: n_tuple[2])
