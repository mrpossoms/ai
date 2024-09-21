#include "env.hpp"
#include "policy.hpp"

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

void check_policy_probabilities_discrete()
{
	policy::Discrete policy;
	Environment env;
	trajectory::Trajectory traj(1, policy.observation_size(), policy.action_size());
	assert(near(policy.act(env, traj).sum().item<float>(), 1.f));
}

void check_policy_probabilities_continuous()
{
	policy::Continuous policy;
	Environment env;
	trajectory::Trajectory traj(1, policy.observation_size(), policy.action_size());
	auto a = policy.act(env, traj).index({Slice(0, traj.size()), Slice(0, 2)});
	for(unsigned i = 0; i < policy.action_size(); i++)
	{
		float a_i = a.flatten()[i].item<float>();
		std::cout << "a" << std::to_string(i) << ": " << a_i << std::endl;
		assert(a.flatten()[i].item<float>() < 1.f && a.flatten()[i].item<float>() > 0.f);
	}
}

int main(int argc, char const *argv[])
{
	check_policy_probabilities_discrete();
	check_policy_probabilities_continuous();
	check_positive_reinforcement();

	return 0;
}