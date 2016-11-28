import numpy as np
import math
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

    def jostle(self, force=0.25):
        assert(force > 0 and force < 1)
        noise = np.random.random(self.soln.shape)
        self.soln = self.soln * (1 - force) + noise * force

    def save(self):
        np.savetxt(fname=self.name, X=self.soln, delimiter=',')

def compute_gradient(trainer, candidate, col_range, resource):
    #print("I'm a thread", str(candidate.soln.shape[1]), col_range)
    for x in col_range:
        for y in range(candidate.soln.shape[1]):
            dxy = candidate.soln.copy()
            dxy[x][y] += trainer.gamma

            # diff, samples = 0, 10
            # for _ in range(samples):
            #     diff += trainer.diff(candidate.soln, dxy, resource) / samples

            diff = trainer.diff(candidate.soln, dxy, resource)

            # print("diff (%d, %d) = %f" % (x, y, diff))
            candidate.g_soln_space[x][y] = (trainer.gamma * diff) / trainer.gamma

class Trainer:
    def __init__(self, maximizing=True):
        self.maximizing = maximizing
        self.gamma = 1 # learning rate
        self.thread_count = 1
        self.os = ObservationSpace(0.01)
        print('Trainer initialized', self.os)

        def subprocess_resource(self, subproc_idx):
            raise NotImplemented

    def train(self, iterations, candidate, target_score=None):
        col_per_thread = math.ceil(candidate.soln.shape[0] / self.thread_count)
        last_score = None
        learning_rate = 0

        for i in range(iterations):
            # reset the observation-space table
            self.os.clear()
            threads = []

            # initialize threads, providing them with the needed resources
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

            if last_score is None:
                last_score = score

            learning_rate = (score - last_score) * 0.6 + learning_rate * 0.3
            last_score = score
            print("score %f, learning rate %f" % (score, learning_rate))

            # if we are stuck in a local maximum. Reset the candidate matrix
            if learning_rate < 0:
                print("Stuck, nudging")
                candidate.jostle(force=0.75)

            if target_score is not None:
                if self.maximizing and score >= target_score:
                    return
                elif not self.maximizing and score <= target_score:
                    return

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
            if nearest is not None:
                old_start, old_score, distance = nearest

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
        nearest = None

        self.lock.acquire()
        for (observation, score) in self.observations:
            dist = np.linalg.norm(key - observation)
            if nearest is None or (dist < self.threshold and dist < nearest[2]):
                nearest = (observation, score, dist)
        self.lock.release()

        if nearest[2] > self.threshold:
            nearest = None

        return nearest
