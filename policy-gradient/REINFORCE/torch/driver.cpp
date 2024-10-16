#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <random>

#include "policy.hpp"
#include "env.hpp"

static Environment env;

using Policy = policy::Continuous;

std::shared_ptr<trajectory::Trajectory> traj;
std::shared_ptr<Policy> P;

int playing()
{
	// return 0 when the game loop should terminate
	return PLAYING;
}

unsigned episode = 0;
float rewards = 0;
bool policy_loaded;

void update()
{
	// sim_step();
	P->act(env, *traj);

	if (policy_loaded)
	{
		TG_TIMEOUT = 10000;
		if (env.distance_to_goal() < 2 || traj->size() >= 64)
		{
			env.spawn(env.state.goal);
			traj->clear();
		}
	}
	else
	{
		if (traj->size() >= 64)
		{
			rewards += traj->R();

			if (episode % 100 == 0)
			{
				std::cout << rewards / 1000.f << " ========================" << std::endl;
				rewards = 0;
			}

			if (episode % 1000 == 0 && episode / 1000 >= 1)
			{
				// policy::save("model.pt");
				torch::save(P, "model.pt");
			}

			// policy::train_policy_gradient(traj, policy::hyper_parameters{(unsigned)traj->size(), 0, 0.001});
			P->train(*traj, 0.01f);
			episode++;

			env.reset();
			traj->clear();
		}
	}
}


int main(int argc, char* argv[])
{
	torch::manual_seed(0);
	// policy::init(4, 4);
	P = std::make_shared<Policy>();
	traj = std::make_shared<trajectory::Trajectory>(64, P->observation_size(), P->action_size(), P->output_size());

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
	unsigned i = 0;

	while (playing())
	{
		update();
	
		// if (policy::loaded())
		if (episode % 1000 == 0 && i > 1000)
		{

			env.render();		
		}
		i++;
	}

	return 1;
}
