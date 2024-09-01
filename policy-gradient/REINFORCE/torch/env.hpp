#pragma once
#include <unistd.h>
#include <vector>
// #include "rl.hpp"

extern int TG_TIMEOUT;
extern bool PLAYING;

struct Environment
{
	struct State
	{
		float position[2];
		float vel[2];
		float goal[2];
		float last_reward;

		int trace[40][80] = {};
	};

	size_t rows();

	size_t cols();

	Environment();
	~Environment();

	State state;

	void reset();

	const std::vector<float>& get_state_vector();

	float distance_to_goal();

	void spawn(float p[2]);

	float step_reward(float u[2]);

	void render();

private:
	std::vector<float> _state_vector;
};
