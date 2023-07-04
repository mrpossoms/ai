#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <curses.h>
#include <term.h>
#include <termios.h>
#include <time.h>
#include <math.h>

#include "spare-time/tg.h"
#include "structs.hpp"
#include "net.hpp"
#include "rl.hpp"

int TG_TIMEOUT = 100000;

struct {
	float position[2];
	float speed = 0;
	float front[2];
	float heading;
	float goal[2];
} ENV;

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
			ENV.position[0] += ENV.front[0] - ENV.position[0];
			ENV.position[1] += ENV.front[1] - ENV.position[1];
			break;
		case 'k':
			ENV.position[0] -= ENV.front[0] - ENV.position[0];
			ENV.position[1] -= ENV.front[1] - ENV.position[1];
			break;
		case 'j':
			ENV.heading -= M_PI / 4;
			break;
		case 'l':
			ENV.heading += M_PI / 4;
			break;
		default:
			// TODO
			;
	}
}


static inline const char* sampler(int row, int col)
{
	if (row == round(ENV.front[0]) && col == round(ENV.front[1]))
	{
		// auto i = static_cast<unsigned>(round(8 * ENV.heading / (M_PI * 2))) % 8;
		// static char out[2];
		// out[0] = "-\\|/-\\|/"[i];
		return ".";
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

void spawn_goal()
{
	ENV.goal[0] = rand() % term.max_rows;
	ENV.goal[1] = rand() % term.max_cols;
}


int playing()
{
	// return 0 when the game loop should terminate
	return 1;
}

float distance()
{
	auto dx = ENV.goal[0] - ENV.position[0];
	auto dy = ENV.goal[1] - ENV.position[1];
	return sqrt(dx*dx + dy*dy);
}

state get_state()
{
	auto dx = ENV.goal[0] - ENV.position[0];
	auto dy = ENV.goal[1] - ENV.position[1];
	return {
		{dx, dy},
		{sin(ENV.heading),cos(ENV.heading)}
	};
}

void update()
{
	auto d_t_1 = distance();
	ENV.front[0] = 1 * sin(ENV.heading) + ENV.position[0];
	ENV.front[1] = 1 * cos(ENV.heading) + ENV.position[1];
	ENV.position[0] += (ENV.front[0] - ENV.position[0]) * ENV.speed;
	ENV.position[1] += (ENV.front[1] - ENV.position[1]) * ENV.speed;
	auto d_t = distance();

	auto reward_t = d_t - d_t_1;

	auto a = net::act(get_state());
	ENV.heading += a.d_heading;
	ENV.speed = std::max(0.5f, std::min(-0.5f, a.d_pos));
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

	spawn_goal();

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
