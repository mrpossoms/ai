import numpy as np
import os

TRAINING_EPOCHS = 200
GENERATION_SIZE = 25
RANDOMIZATION = 1
STAGNATION_THRESHOLD = 5


def rnd_white(base, scale=1, std_dev=1):
    matrix = (np.random.random(base.shape) * 2 - 1) * scale

    if base is not None:
        matrix = matrix + base

    return matrix


def rnd_gaussian(base, scale=1, std_dev=1):
    matrix = np.random.normal(0, std_dev, base.shape) * scale

    if base is not None:
        matrix = matrix + base

    return matrix


class Agent:
    def __init__(self, env, target, reset=True):
        self.environment = env
        self.improvement_rate = 0
        self.target = target
        self.learning_rate = 1

        env.reset()
        observation, reward, done, info = env.step(0)
        rows = len(observation)
        columns = env.action_space.n

        self.shape = (rows, columns)
        self.best_candidate = np.random.random(self.shape)

        if reset:
            try:
                os.unlink("learning.csv")
            except:
                pass

        try:
            self.best_candidate = np.loadtxt(fname='params.txt', delimiter=",")
            print("Loaded existing params")
        except:
            pass

    def gen_candidate(self, randomizer=rnd_gaussian, std_dev=1, scale=1):
        return randomizer(base=self.best_candidate, std_dev=std_dev, scale=scale)

    def run_epoch(self, candidate, show=False):
        avg_reward = 0

        for trial in range(4):
            self.environment.reset()
            observation, reward, done, info = self.environment.step(0)
            steps = 200
            if show:
                steps += 200

            for _ in range(steps):
                observation = np.array(observation)
                action_probs = observation.dot(candidate)

                if show:
                    self.environment.render()

                observation, reward, done, info = self.environment.step(np.argmax(action_probs))
                avg_reward += reward

                if done:
                    break

        if show:
            print(avg_reward / 4)

        return avg_reward / 4

    def evaluate_generation(self, base):
        t_candidate_score = []
        avg_score = 0

        # score the existing candidates
        for idx in range(GENERATION_SIZE):
            candidate = None

            if idx < len(base):
                candidate = base[idx]
            else:
                candidate = rnd_gaussian(base=base[idx % len(base)], std_dev=self.learning_rate, scale=self.learning_rate)

            score = self.run_epoch(candidate=candidate)
            avg_score += score / float(GENERATION_SIZE)

            t_candidate_score += [(candidate, score)]

        t_candidate_score = sorted(t_candidate_score, key=lambda cs_tuple: cs_tuple[1])

        assert(t_candidate_score[-1][1] > t_candidate_score[0][1])

        return t_candidate_score.pop(), avg_score

    def solve(self):
        first_score = None
        last_gen_avg = -600
        last_best_candidates = [[self.best_candidate, last_gen_avg]]

        for i in range(TRAINING_EPOCHS):

            # # re score all the best old candidates
            # for old_best in last_best_candidates:
            #     score = self.run_epoch(candidate=old_best[0])
            #     old_best[1] = (old_best[1] + score) / 2

            # evaluate the generation based off of the old bests
            (best_candidate, best_score), avg_score = self.evaluate_generation(base=list(map(lambda c: c[0], last_best_candidates)))

            # evaluate how much improvment has taken place between generations
            iteration_improvement = avg_score - last_gen_avg
            self.improvement_rate = self.improvement_rate * .75 + iteration_improvement * .25

            # set the 'first score' which dictates with how much variation random candidates are generated
            # if improvement stagnates, this is reset as if learning just began
            if first_score is None:
                first_score = avg_score
                last_gen_avg = avg_score

            if self.improvement_rate < STAGNATION_THRESHOLD:
                first_score = first_score * 0.75 + avg_score * 0.25

            self.learning_rate = (1 - max(0.0001, (last_gen_avg - first_score) / (self.target - first_score)))
            last_gen_avg = avg_score

            print("(%d/%d) Best score: %f Avg: %f Improvement: %f ∆I: %f  ∑: %f" %
                  (i, TRAINING_EPOCHS, best_score, avg_score, iteration_improvement, self.improvement_rate, self.learning_rate))

            # are we done???
            if avg_score - first_score >= self.target - first_score:
                break

            # trim up
            last_best_candidates += [[best_candidate.copy(), best_score + avg_score]]
            last_best_candidates = sorted(last_best_candidates, key=lambda cs_tuple: cs_tuple[1])
            if len(last_best_candidates) > 10:
                last_best_candidates = last_best_candidates[1:len(last_best_candidates)]

            self.best_candidate = last_best_candidates[-1][0].copy()
            best_score = last_best_candidates[-1][1]

            # in the event of a regression, use the best from the last improvement
            if self.improvement_rate < 0:
                print("!!!Regression!!!")

            # save current best candidate
            np.savetxt(fname='params.txt', X=best_candidate, delimiter=',')

            # save learning profile
            with open("learning.csv", mode="a+") as history:
                # matrix_elements = ""
                # reshaped = best_candidate.reshape(best_candidate.size)
                # for element in reshaped:
                #     matrix_elements += str(element) + ','

                history.write(
                    "{0},{1},{2},{3}\n".format(str(best_score), str(avg_score), str(self.improvement_rate), str(self.learning_rate)))
