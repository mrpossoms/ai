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
		struct Frame
		{
			torch::Tensor state;
			torch::Tensor action_probs;
			// torch::Tensor action;
			unsigned action_idx;
			float reward;
		};

		static float R(const std::vector<Frame>& T, float gamma=0.999f);
	};

}