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
	// RL::Action last_action;
	// torch::Tensor last_action_probs;
	// torch::Tensor last_state;
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
	ENV.state.vel[0] = randf() * 0;
	ENV.state.vel[1] = randf() * 0;

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
	return sqrt((dx*dx + dy*dy) + 0.0001);
}

float distance_to_goal()
{
	return distance(ENV.state.position, ENV.state.goal);
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

torch::Tensor get_state_tensor()
{
	auto s = get_state();
	auto x = torch::zeros({1, 4});
	x[0][0] = s.d_goal[0];
	x[0][1] = s.d_goal[1];
	x[0][2] = s.vel[0];
	x[0][3] = s.vel[1];


	// auto options = torch::TensorOptions().dtype(torch::kFloat32);
	// torch::Tensor _s({
	// 	{s.d_goal[0], s.d_goal[1], s.vel[0], s.vel[1]}
	// });

	// return torch::from_blob(s.d_goal, {1, 4}, options);
	return x;
}

void sim_step()
{
	auto x = get_state_tensor();
	auto a_probs = net::act_probs(x).detach(); //.perturb(1.0f, 0.1f));
	// std::cout << "x: " << x << " | u: " << a_probs << std::endl;
	auto a = net::act(a_probs);

	const auto k_speed = 0.1f;

	auto max_idx = torch::argmax(a_probs).item<int>();

	switch(max_idx)
	{
		case 0: ENV.state.vel[0] += k_speed; break;
		case 1: ENV.state.vel[0] += -k_speed; break;
		case 2: ENV.state.vel[1] += k_speed; break;
		case 3: ENV.state.vel[1] += -k_speed; break;
	}

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
	auto reward_t = (d_t_1 - d_t);// - 0.1f;

	if (distance_to_goal() < 2)
	{
		reward_t += 1.f;
	}

	assert(std::isnan(reward_t) == 0);

	ENV.state.last_reward = reward_t;

	traj.append(x, a_probs, reward_t);
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
