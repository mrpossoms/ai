#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <random>

#include "policy.hpp"
#include "env.hpp"

Environment env;

std::vector<Trajectory::Frame> traj;
std::shared_ptr<policy::Discrete> P;

int playing()
{
	// return 0 when the game loop should terminate
	return PLAYING;
}

void sim_step()
{
	auto x_vec = env.get_state_vector();
	auto x = torch::from_blob(x_vec.data(), {1, (long)x_vec.size()}, torch::kFloat);
	auto a_probs = policy::act_probs(x); //.perturb(1.0f, 0.1f));
	// std::cout << a_probs << std::endl;
	auto a = policy::act(a_probs, true);

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
bool policy_loaded;

void update()
{
	// sim_step();
	P->act(env.get_state_vector(), env, traj);

	if (policy_loaded)
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
				// policy::save("model.pt");
				torch::save(P, "model.pt");
			}

			// policy::train_policy_gradient(traj, policy::hyper_parameters{(unsigned)traj.size(), 0, 0.001});
			P->train(traj, 0.01f);
			episode++;

			env.reset();
			traj.clear();
		}
	}
}


int main(int argc, char* argv[])
{
	// policy::init(4, 4);
	P = std::make_shared<policy::Discrete>(4, 4);

	try
	{
		torch::load(P, std::string("model.pt"));
		policy_loaded = true;
	}
	catch (const c10::Error& e)
	{
		policy_loaded = false;
	}

	env.reset();

	while (playing())
	{
		update();
	
		// if (policy::loaded())
		if (episode % 1000 == 0)
		{

			env.render();		
		}
	}

	return 1;
}
