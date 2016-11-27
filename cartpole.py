#!/usr/bin/python
import gym
import numpy as np
import trainer

class CartTrainer(trainer.Trainer):
    def __init__(self):
        trainer.Trainer.__init__(self)
        self.envs = {}

    def subprocess_resource(self, subproc_idx):
        env = None
        key = (subproc_idx)
        if key in self.envs:
            env = self.envs[key]
        else:
            env = self.envs[key] = gym.make('CartPole-v0')

        env.idx = subproc_idx
        return env

    def evaluate(self, candidate, env, show=False):
        total_score = 0
        start = last = env.reset()

        for _ in range(200):
            action = np.argmax(np.array(last).dot(candidate))
            obs, reward, done, info = env.step(action)
            total_score += reward

            if show:
                env.render()

            if done:
                break

        return start, total_score

env = gym.make("CartPole-v0")
cart_agent = trainer.Subject(env.reset(), path="cart_params.txt")

trainer = CartTrainer()

trainer.train(100, cart_agent)
trainer.evaluate(cart_agent.soln, env, show=True)
