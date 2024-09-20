#include "env.hpp"

#include <math.h>
#include <stdint.h>
#include "tg.h"

#include <algorithm>
#include <random>
#include <cassert>
#include <signal.h>
#include <curses.h>
#include <term.h>
#include <termios.h>
#include <time.h>

static std::random_device rd;  // a seed source for the random number engine
static std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()

int TG_TIMEOUT = 100;

struct {
	int max_rows, max_cols;
} term = { 18, 0 };

struct termios oldt;
bool PLAYING = true;

Environment::State* G_STATE;

void sig_winch_hndlr(int sig)
{
	term.max_cols = tg_term_width();
	term.max_rows = tg_term_height();

	term.max_cols = std::min(term.max_cols, 80);
	term.max_rows = std::min(term.max_rows, 40);
}


void sig_int_hndlr(int sig)
{
	tg_restore_settings(&oldt);
	PLAYING = false;
}

static float randf()
{
	return (float)(((int)rand() % 2048) - 1024) / 1024.0f;
}

void input_hndlr()
{
	char c;
	if (tg_key_get(&c) == 0)
	{ // no key pressed
		return;
	}

	switch(c)
	{ // handle key accordingly
		case 'i':
			G_STATE->vel[0] -= 0.5f;
			break;
		case 'k':
			G_STATE->vel[0] += 0.5f;
			break;
		case 'j':
			G_STATE->vel[1] -= 0.5f;
			break;
		case 'l':
			G_STATE->vel[0] += 0.5f;
			break;
		default:
			// TODO
			;
	}
}


static inline const char* sampler(int row, int col)
{
	if (row == (int)(G_STATE->position[0]) && col == (int)(G_STATE->position[1]))
	{
		if (G_STATE->last_reward > 0)
		{
			return "\033[0;32mo\033[0m";
		}
		else if (G_STATE->last_reward < 0)
		{
			return "\033[0;31mo\033[0m";
		}
		else
		{
			return "o";
		}
	}

	if (row == (int)(G_STATE->goal[0]) && col == (int)(G_STATE->goal[1]))
	{
		return "X";
	}

	// return character for a given row and column in the terminal
	static char tmp[2] = {};
	char spec[] = " .,'\"*";

	G_STATE->trace[row][col] = std::max(G_STATE->trace[row][col]-1, 0);

	tmp[0] = spec[(5 * G_STATE->trace[row][col]) / 100];
	return tmp;
}

float distance(float p0[2], float p2[2])
{
	auto dx = p0[0] - p2[0];
	auto dy = p0[1] - p2[1];
	return sqrt((dx*dx + dy*dy) + 0.0001);
}

Environment::Environment() : _state_vector(4)
{
	signal(SIGWINCH, sig_winch_hndlr);
	signal(SIGINT, sig_int_hndlr);
	sig_winch_hndlr(0);

	tg_game_settings(&oldt);

	term.max_rows = 40;
	term.max_cols = 80;
}

Environment::~Environment()
{
	tg_restore_settings(&oldt);
}

float Environment::distance_to_goal()
{
	return distance(state.position, state.goal);
}

size_t Environment::rows() { return term.max_rows; }
size_t Environment::cols() { return term.max_cols; }

void Environment::spawn(float p[2])
{
    std::uniform_int_distribution<> r_dist(1, term.max_rows-1);
    std::uniform_int_distribution<> c_dist(0, term.max_cols-1);

	p[0] = r_dist(gen);
	p[1] = c_dist(gen);	
}

void Environment::reset()
{
	// srand(0);
	state.goal[0] = 10;
	state.goal[1] = 12;
	state.position[0] = 40;
	state.position[1] = 21;
	spawn(state.goal);
	spawn(state.position);
	state.vel[0] = randf() * 0;
	state.vel[1] = randf() * 0;

	memset(state.trace, 0, sizeof(state.trace));
}

const std::vector<float>& Environment::get_state_vector()
{
	auto dx = state.goal[0] - state.position[0];
	auto dy = state.goal[1] - state.position[1];

	_state_vector[0] = dx;
	_state_vector[1] = dy;
	_state_vector[2] = state.vel[0];
	_state_vector[3] = state.vel[1];

	return _state_vector;
}

float Environment::step_reward(float u[2])
{
	assert(!std::isnan(u[0]));
	assert(!std::isnan(u[1]));

	// clamp u
	float u0 = std::max(-0.1f, std::min(0.1f, u[0]));
	float u1 = std::max(-0.1f, std::min(0.1f, u[1]));



	state.vel[0] += u0;
	state.vel[1] += u1;

	auto d_t_1 = distance_to_goal();

	// Simulate movement dynamics
	state.position[0] += state.vel[0];
	state.position[1] += state.vel[1];

	if (state.position[0] > 0 && state.position[0] < rows() && state.position[1] >= 0 && state.position[1] < cols())
	{
		state.trace[(int)state.position[0]][(int)state.position[1]] = 100;
	}

	state.vel[0] *= 0.95f;
	state.vel[1] *= 0.95f;
	auto d_t = distance_to_goal();

	// compute reward
	float zero[2] = {0, 0};
	auto reward_t = (d_t_1 - d_t);// - 0.1f;

	if (distance_to_goal() < 2)
	{
		reward_t += 1.f;
	}

	assert(std::isnan(reward_t) == 0);

	state.last_reward = reward_t;

	return reward_t;
}

void Environment::render()
{
	G_STATE = &state;
	input_hndlr();
	tg_rasterize(term.max_rows, term.max_cols, sampler);
	tg_clear(term.max_rows);
}