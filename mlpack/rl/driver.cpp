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

int TG_TIMEOUT = 10000;

struct {
	float position[2];
	float vel[2];
	float goal[2];
} ENV;

RL::ReplayBuffer replay_buffer(1);

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
	if (row == 0 && col == 0)
	{
		static char buf[32];
		auto& traj = replay_buffer.current_trajectory();
		snprintf(buf, sizeof(buf), "%d/%lld - %d - %f", traj.write_ptr, traj.states.n_cols, replay_buffer.write_ptr, traj.R());
		return buf;
	}

	if (row == round(ENV.goal[0]) && col == round(ENV.goal[1]))
	{
		return "X";
	}

	if (row == round(ENV.position[0]) && col == round(ENV.position[1]))
	{
		return "o";
	}

	// return character for a given row and column in the terminal
	return " ";
}

void spawn(float p[2])
{
	p[0] = 1 + rand() % (term.max_rows-1);
	p[1] = rand() % term.max_cols;	
}

void reset()
{
	// srand(0);
	ENV.goal[0] = 10;
	ENV.goal[1] = 12;
	ENV.position[0] = 40;
	ENV.position[1] = 21;
	// spawn(ENV.goal);
	// spawn(ENV.position);
	ENV.vel[0] = 0;
	ENV.vel[1] = 0;
}

float randf()
{
	return (float)(rand() % 2048 - 1024) / 1024.0f;
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
	return distance(ENV.vel, ENV.goal) + 0.0001;
}

RL::State get_state()
{
	auto dx = ENV.goal[0] - ENV.position[0];
	auto dy = ENV.goal[1] - ENV.position[1];
	return {
		{dx, dy},
		{ENV.vel[0], ENV.vel[1]}
	};
}

void sim_step()
{
	auto d_t_1 = distance_to_goal();
	ENV.position[0] += ENV.vel[0];
	ENV.position[1] += ENV.vel[1];

	if (ENV.position[0] < 1)
	{
		ENV.position[0] = 1;
		ENV.vel[0] = 0;
	}
	if (ENV.position[0] >= term.max_rows)
	{
		ENV.position[0] = term.max_rows-1;
		ENV.vel[0] = 0;
	}
	if (ENV.position[1] < 0)
	{
		ENV.position[1] = 0;
		ENV.vel[1] = 0;
	}
	if (ENV.position[1] >= term.max_cols)
	{
		ENV.position[1] = term.max_cols-1;
		ENV.vel[1] = 0;
	}


	ENV.vel[0] *= 0.9f;
	ENV.vel[1] *= 0.9f;
	auto d_t = distance_to_goal();

	auto reward_t = (d_t_1 - d_t) - 0.0001f;

	if (distance_to_goal() < 4)
	{
		reward_t += 10;
	}

	auto a = net::act(get_state());
	a.d_r += randf() * 0.1f;
	a.d_c += randf() * 0.1f;
	ENV.vel[0] += std::min(0.1f, std::max(-0.1f, a.d_r));
	ENV.vel[1] += std::min(0.1f, std::max(-0.1f, a.d_c));

	replay_buffer.append(get_state(), a, reward_t);

}

void update()
{
	sim_step();

	if (replay_buffer.full())
	{
		std::cout << replay_buffer.avg_reward() << std::endl;
		for (auto& t : replay_buffer.trajectories)
		{
			net::train_policy_gradient(t, {replay_buffer.current_trajectory().states.n_cols, 0, 0.01f});
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

	net::init(4, 2);

	reset();

	while (playing())
	{
		input_hndlr();
		update();
		tg_rasterize(term.max_rows, term.max_cols, sampler);
		tg_clear(term.max_rows);
	}

	tg_restore_settings(&oldt);

	return 1;
}
