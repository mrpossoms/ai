import gym
from agent import Agent
from getopt import getopt
import sys

env = gym.make('LunarLander-v2')
agent = Agent(env)

if not '--just-show' in sys.argv:
	print("Solving")
	agent.solve()

agent.run_epoch(show=True)