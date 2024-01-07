#include "renderer.hpp"

#include <math.h>
#include <stdint.h>
#include "tg.h"

#include <algorithm>
#include <signal.h>
#include <curses.h>
#include <term.h>
#include <termios.h>
#include <time.h>


int TG_TIMEOUT = 100;

struct {
	int max_rows, max_cols;
} term = { 18, 0 };

struct termios oldt;
bool PLAYING = true;

Renderer::State* G_STATE;

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

Renderer::Renderer()
{
	signal(SIGWINCH, sig_winch_hndlr);
	signal(SIGINT, sig_int_hndlr);
	sig_winch_hndlr(0);

	tg_game_settings(&oldt);

	term.max_rows = 40;
	term.max_cols = 80;
}

Renderer::~Renderer()
{
	tg_restore_settings(&oldt);
}

size_t Renderer::rows() { return term.max_rows; }
size_t Renderer::cols() { return term.max_cols; }

void Renderer::render(State& state)
{
	G_STATE = &state;
	input_hndlr();
	tg_rasterize(term.max_rows, term.max_cols, sampler);
	tg_clear(term.max_rows);
}