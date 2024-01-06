#pragma once
#include "torch/torch.h"

static float randf()
{
	return (float)(((int)rand() % 2048) - 1024) / 1024.0f;
}

namespace RL 
{
	union State
	{
		float x[4];
		struct
		{
			float d_goal[2];
			float vel[2];
		};
	};

	union Action
	{
		float u[4];
		struct
		{
			float d_r_pos;
			float d_r_neg;
			float d_c_pos;
			float d_c_neg;
		};
	};


	struct Trajectory
	{
		std::vector<torch::Tensor> action_probs;
		std::vector<float> rewards;

		Trajectory(size_t len=128);

		void append(const State& x, torch::Tensor& action_probs, float r);

		float R(float gamma=0.999f);

		bool full() const { return action_probs.size() >= action_probs.capacity(); }

		void clear();
	};

}