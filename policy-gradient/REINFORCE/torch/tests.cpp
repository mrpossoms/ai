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
	for (unsigned e = 0; e < 100; e++)
	{
		policy::Continuous policy;
		Environment env;
		trajectory::Trajectory traj(1, policy.observation_size(), policy.action_size());
		
		auto a = policy.act(env, traj); //.index({Slice(0, traj.size()), Slice(0, 2)});
		for(unsigned i = 0; i < policy.action_size(); i++)
		{
			float a_i = a.flatten()[i].item<float>();
			std::cout << "a" << std::to_string(i) << ": " << a_i << std::endl;
			assert(a.flatten()[i].item<float>() < 1.f && a.flatten()[i].item<float>() > 0.f);
		}		
	}
}

void check_policy_optimization_continuous()
{
	policy::Continuous policy;
	Environment env;
	trajectory::Trajectory traj(1, policy.observation_size(), policy.action_size());

	std::vector<float> probabilities;
	std::vector<float> sigmas;

	for (int i = 0; i < 100; i++)
	{
		auto x = torch::ones({1, 4});
		auto y = policy.forward(x);
		auto a = torch::ones({1, 2});
		auto r = torch::ones({1});
		auto pr = policy.action_probabilities(y, a);
		auto sigma = policy.action_sigma(y);
		auto mu = y.index({0, Slice(0, 2)});
		probabilities.push_back(pr[0][0].item<float>());
		sigmas.push_back(sigma[0].item<float>());
		std::cout << 
		"pr" << std::to_string(i) << ": " << pr[0][0].item<float>() << 
		" mu: " << mu[0].item<float>() << 
		" sig: " << sigmas[sigmas.size()-1] << std::endl;
		traj.push_back({x, pr, a, 0, r});
		policy.train(traj, 0.1f);
		traj.clear();
	}

	for (int i = 1; i < probabilities.size(); i++)
	{
		assert(probabilities[i] > probabilities[i - 1]);
	}

	// auto a = policy.act(env, traj).index({Slice(0, traj.size()), Slice(0, 2)});
	// auto old_probs = a.clone();
	// auto reward = torch::ones({1, 1});
	// auto reward_t = torch::from_blob(&reward, {1}, torch::kFloat).clone();
	// traj.push_back({torch::ones({1, 1}), a, a, 0, reward_t});
	// policy.train(traj, 0.1f);
	// auto new_probs = policy.act(env, traj).index({Slice(0, traj.size()), Slice(0, 2)});
	// assert((new_probs - old_probs).sum().item<float>() > 0);
}

int main(int argc, char const *argv[])
{
	check_policy_probabilities_discrete();
	check_policy_probabilities_continuous();
	check_positive_reinforcement();
	check_policy_optimization_continuous();
	
	return 0;
}
