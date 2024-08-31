#pragma once
#include <unistd.h>
#include <tuple>
#include <vector>

#include "torch/torch.h"

struct Trajectory
{
	struct Frame
	{
		torch::Tensor state;
		torch::Tensor action_probs;
		// torch::Tensor action;
		unsigned action_idx;
		float reward;
	};

	static float R(const std::vector<Frame>& T, float gamma=0.999f)
	{
		float r = 0.0f;
		for (int i = 0; i < T.size(); i++)
		{
			r += pow(gamma, i) * T[i].reward;
		}

		return r;
	}
};

namespace net 
{
	struct hyper_parameters
	{
		unsigned epochs = 1;
		unsigned batch_size = 1;
		float learning_rate = 0.001f;
	};

	void init(size_t observation_size, size_t action_size);

	bool loaded();

	void save(const std::string& path);

	void train_policy_gradient(const std::vector<Trajectory::Frame>& traj, const hyper_parameters& hp={});

	torch::Tensor act_probs(torch::Tensor x);

	int act(torch::Tensor probs, bool stochastic=true);

}
