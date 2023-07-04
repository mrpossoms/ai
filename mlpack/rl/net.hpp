#pragma once
#include <tuple>
#include <vector>
#include "structs.hpp"

namespace net 
{
	struct hyper_parameters
	{
		unsigned epochs = 1;
		unsigned batch_size = 128;
		float learning_rate = 0.001f;
	};

	void init(size_t observation_size, size_t action_size);

	void train_policy_gradient(const std::vector<std::tuple<state, action, float>>& trajectory, hyper_parameters& hp);

	action act(const state& x);
}
