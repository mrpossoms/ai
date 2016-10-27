import gym
from agent import Agent
from getopt import getopt
import sys

just_show = '--just-show' in sys.argv

env = gym.make('LunarLander-v2')
agent = Agent(env, target=200, reset=not just_show)

if not just_show:
	agent.solve()

agent.run_epoch(show=True, candidate=agent.best_candidate)