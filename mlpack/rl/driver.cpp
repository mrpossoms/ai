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

int TG_TIMEOUT = 100;
bool PLAYING = true;

struct {
	float position[2];
	float vel[2];
	float goal[2];
	float last_reward;
	RL::Action last_action;

	int trace[40][80] = {};
} ENV;

RL::ReplayBuffer replay_buffer(64);

struct {
	int max_rows, max_cols;
} term = { 18, 0 };

struct termios oldt;
static std::random_device rd;  // a seed source for the random number engine
static std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()

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
	char spec[] = " .,'\"*";

	ENV.trace[row][col] = std::max(ENV.trace[row][col]-1, 0);

	tmp[0] = spec[(5 * ENV.trace[row][col]) / 100];
	return tmp;
}

void spawn(float p[2])
{
    std::uniform_int_distribution<> r_dist(1, term.max_rows-1);
    std::uniform_int_distribution<> c_dist(0, term.max_cols-1);

	p[0] = r_dist(gen);
	p[1] = c_dist(gen);	
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

	memset(ENV.trace, 0, sizeof(ENV.trace));
}

int playing()
{
	// return 0 when the game loop should terminate
	return PLAYING;
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
	};
}

void sim_step()
{
	auto d_t_1 = distance_to_goal();
	ENV.position[0] += ENV.vel[0];
	ENV.position[1] += ENV.vel[1];

	if (ENV.position[0] > 0 && ENV.position[0] < term.max_rows && ENV.position[1] >= 0 && ENV.position[1] < term.max_cols)
	{
		ENV.trace[(int)ENV.position[0]][(int)ENV.position[1]] = 100;
	}

	ENV.vel[0] *= 0.9f;
	ENV.vel[1] *= 0.9f;
	auto d_t = distance_to_goal();

	float zero[2] = {0, 0};
	auto reward_t = (d_t_1 - d_t) - 0.1f;

	if (distance_to_goal() < 2)
	{
		reward_t += 1.f;
	}

	ENV.last_reward = reward_t;
	replay_buffer.append(get_state(), ENV.last_action, reward_t);

	auto a = net::act(get_state()); //.perturb(1.0f, 0.1f));
	auto u_r = a.d_r_pos;
	auto u_c = a.d_c_pos;

	if (a.d_r_pos < a.d_r_neg)
	{
		u_r = -a.d_r_neg;
	}

	if (a.d_c_pos < a.d_c_neg)
	{
		u_c = -a.d_c_neg;
	}

	ENV.last_action = {u_r, u_c};

	u_r = std::min(0.1f, std::max(-0.1f, u_r));
	u_c = std::min(0.1f, std::max(-0.1f, u_c));

	ENV.vel[0] += u_r;
	ENV.vel[1] += u_c;
}

unsigned training_step = 0;

void update()
{
	sim_step();

	if (net::loaded())
	{
		TG_TIMEOUT = 10000;
		if (distance_to_goal() < 2)
		{
			spawn(ENV.goal);
		}
	}
	else
	{
		if (replay_buffer.full())
		{
			std::cout << replay_buffer.avg_reward() << std::endl;
			for (auto& t : replay_buffer.trajectories)
			{
				net::train_policy_gradient(t, {(unsigned)replay_buffer.current_trajectory().states.n_cols, 0, 0.0001});
				training_step++;
			}
			replay_buffer.clear();
			reset();
		}
		else if (replay_buffer.current_trajectory().full())
		{
			reset();
		}
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

	net::init(4, sizeof(RL::Action) / sizeof(float));

	reset();

	while (playing())
	{
		update();
	
		if (net::loaded())
		{
			input_hndlr();
			tg_rasterize(term.max_rows, term.max_cols, sampler);
			tg_clear(term.max_rows);			
		}

		auto& traj = replay_buffer.current_trajectory();
		// printf("%d/%lld - %f (%f)\n", traj.write_ptr, traj.states.n_cols, ENV.last_reward, traj.R());
	}
	

	tg_restore_settings(&oldt);

	return 1;
}
