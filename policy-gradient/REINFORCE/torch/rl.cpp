#include "rl.hpp"

float RL::Trajectory::R(const std::vector<Frame>& T, float gamma)
{
	float r = 0.0f;
	for (int i = 0; i < T.size(); i++)
	{
		r += pow(gamma, i) * T[i].reward;
	}

	return r;
}