#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <curses.h>
#include <term.h>
#include <termios.h>
#include <time.h>
#include <math.h>

#include "spare-time/tg.h"
#include "net.hpp"
#include "rl.hpp"

int TG_TIMEOUT = 5000;

struct {
	float position[2];
	float vel[2];
	float goal[2];
	float last_reward;
	RL::Action last_action;

	char trace[40][80];
} ENV;

RL::ReplayBuffer replay_buffer(8);

struct {
	int max_rows, max_cols;
} term = { 18, 0 };

struct termios oldt;


void sig_winch_hndlr(int sig)
{
	term.max_cols = tg_term_width();
	if (term.max_cols < 0)
	{
		term.max_cols = 80;
	}
}


void sig_int_hndlr(int sig)
{
	tg_restore_settings(&oldt);
	exit(1);
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
			ENV.vel[0] -= 0.5f;
			break;
		case 'k':
			ENV.vel[0] += 0.5f;
			break;
		case 'j':
			ENV.vel[1] -= 0.5f;
			break;
		case 'l':
			ENV.vel[0] += 0.5f;
			break;
		default:
			// TODO
			;
	}
}


static inline const char* sampler(int row, int col)
{
	// if (row == 0 && col == 0)
	// {
	// 	static char buf[32];
	// 	auto& traj = replay_buffer.current_trajectory();
	// 	snprintf(buf, sizeof(buf), "%d/%lld - %f (%f)", traj.write_ptr, traj.states.n_cols, ENV.last_reward, traj.R());
	// 	return buf;
	// }

	if (row == (int)(ENV.position[0]) && col == (int)(ENV.position[1]))
	{
		if (ENV.last_reward > 0)
		{
			return "\033[0;32mo\033[0m";
		}
		else if (ENV.last_reward < 0)
		{
			return "\033[0;31mo\033[0m";
		}
		else
		{
			return "o";
		}
	}

	if (row == (int)(ENV.goal[0]) && col == (int)(ENV.goal[1]))
	{
		return "X";
	}

	// return character for a given row and column in the terminal
	static char tmp[2] = {};
	tmp[0] = ENV.trace[row][col];
	return tmp;
}

void spawn(float p[2])
{
	p[0] = 1 + rand() % (term.max_rows-1);
	p[1] = rand() % term.max_cols;	
}

float randf()
{
	return (float)(((int)rand() % 2048) - 1024) / 1024.0f;
}

void reset()
{
	// srand(0);
	ENV.goal[0] = 10;
	ENV.goal[1] = 12;
	ENV.position[0] = 40;
	ENV.position[1] = 21;
	spawn(ENV.goal);
	spawn(ENV.position);
	ENV.vel[0] = randf();
	ENV.vel[1] = randf();

	memset(ENV.trace, ' ', sizeof(ENV.trace));
}

int playing()
{
	// return 0 when the game loop should terminate
	return 1;
}

float distance(float p0[2], float p2[2])
{
	auto dx = p0[0] - p2[0];
	auto dy = p0[1] - p2[1];
	return sqrt(dx*dx + dy*dy);
}

float distance_to_goal()
{
	return distance(ENV.position, ENV.goal) + 0.0001;
}

RL::State get_state()
{
	auto dx = ENV.goal[0] - ENV.position[0];
	auto dy = ENV.goal[1] - ENV.position[1];
	return {
		{dx, dy},
		{ENV.vel[0], ENV.vel[1]},
		{pow(ENV.vel[0], 3), pow(ENV.vel[1], 3)},
	};
}

void sim_step()
{
	auto d_t_1 = distance_to_goal();
	ENV.position[0] += ENV.vel[0];
	ENV.position[1] += ENV.vel[1];

	// if (ENV.position[0] < 1)
	// {
	// 	ENV.position[0] = 1;
	// 	ENV.vel[0] = 0;
	// }
	// if (ENV.position[0] >= term.max_rows)
	// {
	// 	ENV.position[0] = term.max_rows-1;
	// 	ENV.vel[0] = 0;
	// }
	// if (ENV.position[1] < 0)
	// {
	// 	ENV.position[1] = 0;
	// 	ENV.vel[1] = 0;
	// }
	// if (ENV.position[1] >= term.max_cols)
	// {
	// 	ENV.position[1] = term.max_cols-1;
	// 	ENV.vel[1] = 0;
	// }
	if (ENV.position[0] > 0 && ENV.position[0] < term.max_rows && ENV.position[1] >= 0 && ENV.position[1] < term.max_cols)
	{
		ENV.trace[(int)ENV.position[0]][(int)ENV.position[1]] = '.';
	}

	ENV.vel[0] *= 0.9f;
	ENV.vel[1] *= 0.9f;
	auto d_t = distance_to_goal();

	float zero[2] = {0, 0};
	auto reward_t = (d_t_1 - d_t) - 0.1f;

	if (distance_to_goal() < 2)
	{
		reward_t += 0.2f;
	}

	ENV.last_reward = reward_t;
	replay_buffer.append(get_state(), ENV.last_action, reward_t);

	auto a = net::act(get_state());
	auto u_r = a.d_r + randf() * 0.01f;
	auto u_c = a.d_c + randf() * 0.01f;
	ENV.last_action = {u_r, u_c};

	u_r = std::min(0.1f, std::max(-0.1f, u_r));
	u_c = std::min(0.1f, std::max(-0.1f, u_c));

	ENV.vel[0] += u_r;
	ENV.vel[1] += u_c;
}

void update()
{
	sim_step();

	if (replay_buffer.full())
	{
		std::cout << replay_buffer.avg_reward() << std::endl;
		for (auto& t : replay_buffer.trajectories)
		{
			net::train_policy_gradient(t, {replay_buffer.current_trajectory().states.n_cols, 0, 0.001f});
		}
		replay_buffer.clear();
		reset();
	}
	else if (replay_buffer.current_trajectory().full())
	{
		reset();
	}
}


int main(int argc, char* argv[])
{
	ENV.position[0] = 20;
	ENV.position[1] = 40;

	signal(SIGWINCH, sig_winch_hndlr);
	signal(SIGINT, sig_int_hndlr);
	sig_winch_hndlr(0);

	tg_game_settings(&oldt);

	term.max_rows = 40;
	term.max_cols = 80;

	net::init(6, 2);

	reset();

	while (playing())
	{
		input_hndlr();
		update();
		tg_rasterize(term.max_rows, term.max_cols, sampler);
		tg_clear(term.max_rows);
	
		auto& traj = replay_buffer.current_trajectory();
		// printf("%d/%lld - %f (%f)\n", traj.write_ptr, traj.states.n_cols, ENV.last_reward, traj.R());
	}
	

	tg_restore_settings(&oldt);

	return 1;
}
