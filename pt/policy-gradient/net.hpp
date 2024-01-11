#pragma once
#include <unistd.h>
#include <tuple>
#include <vector>
#include "rl.hpp"

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

	torch::Tensor policy_loss(const RL::Trajectory& traj);

	void train_policy_gradient(const RL::Trajectory& traj, const hyper_parameters& hp={});

	torch::Tensor act_probs(torch::Tensor x);

	RL::Action act(torch::Tensor probs);
}
