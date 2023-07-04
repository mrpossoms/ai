#pragma once
struct state
{
	float d_goal[2];
	float heading[2];
};

struct action
{
	float d_heading;
	float d_pos;
};
