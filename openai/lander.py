import gym
from agent_0 import Agent
import copy

env = gym.make('LunarLander-v2')
observation = env.reset()
agent = Agent()

action=0
best=None
best_agent=None

epochs = 2000
for e in range(0, epochs):
	observation = env.reset()
	total_reward = 0

	observation, reward, done, info = env.step(0)
	agent.step(observation, reward, false)

	for t in range(0, 1000):

		observation, reward, done, info = env.step(action)

		action = agent.step(observation, reward)
		total_reward += reward

		if done:
			# print("{}/{} Episode finished after {} timesteps with {}".format(e, epochs, t+1, total_reward))
			break

observation = env.reset()
total_reward = 0
for t in range(0, 1000):
	env.render()
	observation, reward, done, info = env.step(action)
	action = best_agent.step(observation, reward)
	total_reward += reward

	if done:
		break
