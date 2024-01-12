#include "env.hpp"
#include "net.hpp"
#include "rl.hpp"

#include <iostream>

struct Dummy : torch::nn::Module
{
	Dummy()
	{
		layer = register_module("layer", torch::nn::Linear(1, 4));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		return layer->forward(x);
	}

	torch::nn::Linear layer = { nullptr };
};

bool near(float a, float b)
{
	return std::fabs(a - b) <= std::numeric_limits<float>::epsilon();
}

void check_positive_reinforcement()
{
	RL::Trajectory traj;
	Dummy dummy_net;

	for (unsigned e = 0; e < 3; e++)
	{
		auto probs = dummy_net.forward(torch::ones({1, 1}));
		probs[0][3] += 2;
		probs = torch::softmax(probs, 1);
		std::cout << "------------" << std::endl;
		traj.append({}, probs, 1);

		auto loss = net::policy_loss(traj);
		auto old_probs = probs.clone();
		torch::optim::Adam opt(dummy_net.parameters(), 0.1f);

		opt.zero_grad();
		loss.backward();
		opt.step();
		traj.clear();

		probs = dummy_net.forward(torch::ones({1, 1}));
		probs[0][3] += 2;
		probs = torch::softmax(probs, 1);

		std::cout << "new probs: " << probs << std::endl;
		std::cout << "old probs: " << old_probs << std::endl;		
			
		assert(probs[0][3].item<float>() >= old_probs[0][3].item<float>());
	}
}

void check_policy_probabilities()
{
	net::init(4, 4);
	auto probs = net::act_probs(torch::rand({1, 4}));

	assert(near(probs.sum().item<float>(), 1.f));
}

int main(int argc, char const *argv[])
{
	check_policy_probabilities();
	check_positive_reinforcement();

	return 0;
}