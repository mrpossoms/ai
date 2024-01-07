#pragma once
#include <unistd.h>

extern int TG_TIMEOUT;
extern bool PLAYING;

struct Renderer
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

	Renderer();
	~Renderer();

	void render(State& state);
};