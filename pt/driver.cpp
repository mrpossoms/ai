#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include <random>

#include "net.hpp"
#include "rl.hpp"
#include "renderer.hpp"

static std::random_device rd;  // a seed source for the random number engine
static std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()

struct {
	Renderer::State state;
	RL::Action last_action;
	torch::Tensor last_action_probs;
} ENV;

RL::Trajectory traj(128);
Renderer renderer;

void spawn(float p[2])
{
    std::uniform_int_distribution<> r_dist(1, renderer.rows()-1);
    std::uniform_int_distribution<> c_dist(0, renderer.cols()-1);

	p[0] = r_dist(gen);
	p[1] = c_dist(gen);	
}

void reset()
{
	// srand(0);
	ENV.state.goal[0] = 10;
	ENV.state.goal[1] = 12;
	ENV.state.position[0] = 40;
	ENV.state.position[1] = 21;
	spawn(ENV.state.goal);
	spawn(ENV.state.position);
	ENV.state.vel[0] = randf();
	ENV.state.vel[1] = randf();

	memset(ENV.state.trace, 0, sizeof(ENV.state.trace));
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
	return distance(ENV.state.position, ENV.state.goal) + 0.0001;
}

RL::State get_state()
{
	auto dx = ENV.state.goal[0] - ENV.state.position[0];
	auto dy = ENV.state.goal[1] - ENV.state.position[1];
	return {
		dx, dy,
		ENV.state.vel[0], ENV.state.vel[1],
	};
}

void sim_step()
{
	auto d_t_1 = distance_to_goal();

	// Simulate movement dynamics
	ENV.state.position[0] += ENV.state.vel[0];
	ENV.state.position[1] += ENV.state.vel[1];

	if (ENV.state.position[0] > 0 && ENV.state.position[0] < renderer.rows() && ENV.state.position[1] >= 0 && ENV.state.position[1] < renderer.cols())
	{
		ENV.state.trace[(int)ENV.state.position[0]][(int)ENV.state.position[1]] = 100;
	}

	ENV.state.vel[0] *= 0.9f;
	ENV.state.vel[1] *= 0.9f;
	auto d_t = distance_to_goal();

	// compute reward
	float zero[2] = {0, 0};
	auto reward_t = (d_t_1 - d_t) - 0.1f;

	if (distance_to_goal() < 2)
	{
		reward_t += 1.f;
	}

	ENV.state.last_reward = reward_t;

	// Check if tenso ENV.last_action_probs is initialized
	if (ENV.last_action_probs.numel() != 0)
	{
		traj.append(get_state(), ENV.last_action_probs, reward_t);
	}

	auto a_probs = net::act_probs(get_state()); //.perturb(1.0f, 0.1f));

	auto a = net::act(a_probs);

	auto u_r = a.d_r_pos;
	auto u_c = a.d_c_pos;

	if (a.d_r_pos + a.d_r_neg != 0)
	{
		if (a.d_r_pos < a.d_r_neg)
		{
			u_r = -a.d_r_neg;
		}
		ENV.state.vel[0] += u_r;
	}
	else
	{
		if (a.d_c_pos < a.d_c_neg)
		{
			u_c = -a.d_c_neg;
		}
		ENV.state.vel[1] += u_c;
	}

	ENV.last_action = a;
	ENV.last_action_probs = a_probs;
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
			spawn(ENV.state.goal);
		}
	}
	else
	{
		if (traj.full())
		{
			std::cout << traj.R() << std::endl;
			
			net::train_policy_gradient(traj, net::hyper_parameters{(unsigned)traj.rewards.size(), 0, 0.001});
			training_step++;

			reset();
			traj.clear();
		}
	}
}


int main(int argc, char* argv[])
{
	net::init(4, sizeof(RL::Action) / sizeof(float));

	reset();

	while (playing())
	{
		update();
	
		// if (net::loaded())
		{
			renderer.render(ENV.state);		
		}
	}

	return 1;
}
