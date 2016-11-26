import numpy as np
from threading import Thread
from threading import Lock

class Subject:
	def __init__(self, rep, path=None):
		self.name = path
		
		if path is not None:
			try:
				self.soln = np.loadtxt(fname=path, delimiter=',')
			except:
				self.soln = rep
                else:
			self.soln = rep

                if len(self.soln.shape) < 2:
                    self.soln = self.soln.reshape((self.soln.shape[0], 1))

		self.g_soln_space = np.zeros(self.soln.shape)
		self.d_score = np.zeros(self.soln.shape) 


	def save(self):
		np.savetxt(fname=name, X=self.soln, delimiter=',')

def compute_gradient(trainer, candidate, col_range, resource):
    #print("I'm a thread", str(candidate.soln.shape[1]), col_range)
    for x in col_range: 
        for y in range(candidate.soln.shape[1]):
                dxy = candidate.soln.copy()
                dxy[x][y] += trainer.gamma
                diff = trainer.diff(candidate.soln, dxy, resource)  
                candidate.d_score = diff
                candidate.g_soln_space[x][y] = (trainer.gamma * diff) / trainer.gamma

   
class Trainer:
	def __init__(self, maximizing=True):
                self.maximizing = maximizing
		self.gamma = 0.001 # learning rate
		self.thread_count = 4
                self.os = ObservationSpace(0.25)
                print('Trainer initialized', self.os)

        def subprocess_resource(self, subproc_idx):
            raise NotImplemented

	def train(self, iterations, candidate):
		col_per_thread = candidate.soln.shape[0] / self.thread_count

                for i in range(iterations):
                    # reset the observation-space table
                    self.os.clear()
                    threads = []

                    # initialize threads, providing them with the needed resources
                    # and values to perform gradient computations
                    for i in range(self.thread_count):
                            thread = Thread(target=compute_gradient, kwargs={
                                    "trainer": self,
                                    "candidate": candidate,
                                    "col_range": range(i * col_per_thread, (i + 1) * col_per_thread),
                                    "resource": self.subprocess_resource(subproc_idx=i)
                                })

                            threads += [thread]
                            thread.run()

                    # finish computing gradients
                    for thread in threads:
                        if thread.isAlive():
                            thread.join()
                       
                    # modify solution matrix values by gradient
                    # computed in the previous run
                    candidate.soln += candidate.g_soln_space
                    start, score = self.evaluate(candidate.soln, self.subprocess_resource(0))
                    print("score %f" % score)

        def diff(self, old_candidate, new_candidate, resource):
            #print("observation space:", self.os)
            while True:
                # generate some trial results with the original candidate
                # store them in the observation-space table
                for _ in range(10):
                    start, score = self.evaluate(old_candidate, resource) 
                    self.os + (start, score)

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
        self.lock = Lock()

    def clear(self):
        self.observations = []

    def __add__(self, other):
        self.lock.acquire()
        self.observations.append(other)
        self.lock.release()

    def __getitem__(self, key):
        nearest = []

        self.lock.acquire()
        for (observation, score) in self.observations:
            dist = np.linalg.norm(key - observation)
            if dist < self.threshold:
                nearest += [(observation, score, dist)]
        self.lock.release()

        # ordered from nearest to most distant
        return sorted(nearest, key=lambda n_tuple: n_tuple[2])
