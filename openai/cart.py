# https://gym.openai.com/envs/CartPole-v0

# Code copied and adapted from Siraj's video. Check out his Channel! Thanks Sirajology!
# Sirajology - https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A
# Build a Game Bot (live) - https://www.youtube.com/watch?v=3vxk91K1PiI

import gym
import numpy as np

# hill climbing initialize weights randomly, utilize memory to save food weights

def run_episode(env, params, render=False):
	observation = env.reset()
	totalReward = 0
	t=0
	done = False
	for i in range(1000):
		t = t+1
		if render: env.render()
		action = 0 if np.matmul(params, observation)<0 else 1
		observation, reward, done, info = env.step(action)

		if render: print(observation)

		totalReward += reward
		if done:
			break
	print("Episode finished after {} timesteps".format(t+1))
	return totalReward

# hill climbing
def train():
	noide_scaling = 0.1
	params = np.random.rand(4) * 2 - 1	# [-1,1]
	try:
		params = np.loadtxt("params.txt") # these are getting big! ...
		np.savetxt("params_0.txt", params)
	except Exception as e:
		pass
	print("starting params:")
	print(params)
	bestReward = 0
	rewards = np.array([])
	totalRuns = 1000
	for i_run in range(totalRuns):
		new_params = params + (np.random.rand(4) * 2 - 1) * noide_scaling
		reward = run_episode(env, new_params)

		if reward > bestReward:
			bestReward = reward
			params = new_params
		print("reward %d best %d" % (reward, bestReward))
		rewards = np.append(rewards, reward)	# store rewards for grading

		# show avg of last 100
		avg = np.average(rewards[-100:])
		print("Run: %d of %d AVG: %f" % (i_run, totalRuns, avg))
	print(rewards)
	print(params)
	run_episode(env, params, render=True)
	np.savetxt("params.txt", params)

# main
env = gym.make('CartPole-v0')
#env.monitor.start('/tmp/cartpole-experiment-1', force=True)
r = train()
#env.monitor.close()
print(r)
