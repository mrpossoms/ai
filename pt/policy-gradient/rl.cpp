#include "rl.hpp"

RL::Trajectory::Trajectory(size_t len)
{
	action_probs.reserve(len);
	rewards.reserve(len);
}

void RL::Trajectory::append(torch::Tensor x, torch::Tensor u_probs, float r)
{
	// assert(x.isnan().sum().item<int>() == 0);
	assert(u_probs.isnan().sum().item<int>() == 0);

	state.push_back(x);
	action_probs.push_back(u_probs);
	rewards.push_back(r);
}

float RL::Trajectory::R(float gamma)
{
	float r = 0.0f;
	for (int i = 0; i < rewards.size(); i++)
	{
		r += pow(gamma, i) * rewards[i];
	}

	return r;
}

void RL::Trajectory::clear()
{
	state.clear();
	action_probs.clear();
	rewards.clear();
}