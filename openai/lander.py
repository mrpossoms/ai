import gym
from agent import Agent
from getopt import getopt
import sys

env = gym.make('LunarLander-v2')
agent = Agent(env, target=200)

if not '--just-show' in sys.argv:
	print("Solving")
	agent.solve()

agent.run_epoch(show=True, candidate=agent.best_candidate)