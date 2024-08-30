#include "env.hpp"
#include "net.hpp"
#include "rl.hpp"

#include <iostream>

struct Dummy : torch::nn::Module
{
	Dummy()
	{
		layer = register_module("layer", torch::nn::Linear(1, 2));
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
	Dummy dummy_net;

	auto input = torch::ones({1, 1});
	auto probs = dummy_net.forward(input);
	
	probs = torch::softmax(probs, 1);
	std::cout << "------------" << std::endl;

	auto old_probs = probs.clone();

	probs[0][0].backward();

	for (auto& param : dummy_net.parameters())
	{
		param.data() += param.grad() * 0.1f;
	}

	probs = dummy_net.forward(torch::ones({1, 1}));
	probs = torch::softmax(probs, 1);

	std::cout << "old probs: " << old_probs << std::endl;
	std::cout << "new probs: " << probs << std::endl;
		
	assert(probs[0][0].item<float>() >= old_probs[0][0].item<float>());
}

void check_policy_probabilities()
{
	net::init(4, 4);
	auto probs = net::act_probs(torch::rand({1, 2}));

	assert(near(probs.sum().item<float>(), 1.f));
}

int main(int argc, char const *argv[])
{
	// check_policy_probabilities();
	check_positive_reinforcement();

	return 0;
}