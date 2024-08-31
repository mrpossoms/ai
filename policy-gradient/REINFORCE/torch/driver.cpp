#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <random>

#include "net.hpp"
#include "env.hpp"

Environment env;

std::vector<Trajectory::Frame> traj;

int playing()
{
	// return 0 when the game loop should terminate
	return PLAYING;
}

torch::Tensor get_state()
{
	auto dx = env.state.goal[0] - env.state.position[0];
	auto dy = env.state.goal[1] - env.state.position[1];

	auto x = torch::zeros({1, 4});
	x[0][0] = dx;
	x[0][1] = dy;
	x[0][2] = env.state.vel[0];
	x[0][3] = env.state.vel[1];

	return x;
}

void sim_step()
{
	auto x = get_state();
	auto a_probs = net::act_probs(x); //.perturb(1.0f, 0.1f));
	// std::cout << a_probs << std::endl;
	auto a = net::act(a_probs, true);

	const auto k_speed = 0.1f;

	float u[2] = {};

	switch(a)
	{
		case 0: u[0] += k_speed; break;
		case 1: u[0] += -k_speed; break;
		case 2: u[1] += k_speed; break;
		case 3: u[1] += -k_speed; break;
	}

	auto reward_t = env.step_reward(u);

	traj.push_back({x, a_probs, (unsigned)a, reward_t});
}

unsigned episode = 0;
float rewards = 0;

void update()
{
	sim_step();

	if (net::loaded())
	{
		TG_TIMEOUT = 10000;
		if (env.distance_to_goal() < 2 || traj.size() >= 256)
		{
			env.spawn(env.state.goal);
			traj.clear();
		}
	}
	else
	{
		if (traj.size() >= (episode % 1000 == 0 ? 256 : 128))
		{
			rewards += Trajectory::R(traj);

			if (episode % 100 == 0)
			{
				std::cout << rewards / 1000.f << " ========================" << std::endl;
				rewards = 0;
			}

			if (episode % 1000 == 0)
			{
				net::save("model.pt");
			}

			net::train_policy_gradient(traj, net::hyper_parameters{(unsigned)traj.size(), 0, 0.001});
			episode++;

			env.reset();
			traj.clear();
		}
	}
}


int main(int argc, char* argv[])
{
	net::init(4, 4);

	env.reset();

	while (playing())
	{
		update();
	
		// if (net::loaded())
		if (episode % 1000 == 0)
		{

			env.render();		
		}
	}

	return 1;
}
