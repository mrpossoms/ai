#include "policy.hpp"
#include <random>
// #include "policy_gradient.hpp"
// #define MLPACK_ENABLE_ANN_SERIALIZATION
// #include <mlpack/core.hpp>
// #include <mlpack/methods/ann.hpp>

// using namespace mlpack;


// static FFN<PolicyGradientLoss<arma::mat>, RandomInitialization, arma::mat> model;

// ens::Adam optimizer;


policy::Discrete::Discrete(size_t observation_size, size_t action_size)
{
	l0 = { register_module("l0", torch::nn::Linear(observation_size, 16)) };
	l1 = { register_module("l1", torch::nn::Linear(16, 16)) };
	l2 = { register_module("l2", torch::nn::Linear(16, 8)) };
	l3 = { register_module("l3", torch::nn::Linear(8, action_size)) };
}

torch::Tensor policy::Discrete::forward(torch::Tensor x)
{
	x = torch::leaky_relu(l0->forward(x));
	x = torch::leaky_relu(l1->forward(x));
	x = torch::leaky_relu(l2->forward(x));
	x = torch::softmax(l3->forward(x), 1);

	return x;
}

void policy::Discrete::act(const std::vector<float>& x, Environment& env, std::vector<Trajectory::Frame>& traj)
{
	auto x_t = torch::from_blob((void*)x.data(), {1, (long)x.size()}, torch::kFloat).clone();
	auto a_probs = forward(x_t);
	auto index = torch::multinomial(a_probs, 1);
	auto a = index.item<int>();
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

	traj.push_back({x_t, a_probs, (unsigned)a, reward_t});
}



void policy::Discrete::train(const std::vector<Trajectory::Frame>& traj, float learning_rate)
{
	zero_grad();

	for (unsigned t = 0; t < traj.size(); t++)
	{
		const auto& f_t = traj[t];
		(torch::log(torch::flatten(f_t.action_probs)[f_t.action_idx]) * f_t.reward).backward();
	}

	for (auto& param : parameters())
	{
		param.data() += (param.grad() / static_cast<float>(traj.size())) * learning_rate;
	}
}


policy::Continuous::Continuous(size_t observation_size, size_t action_size)
{
	l0 = { register_module("l0", torch::nn::Linear(observation_size, 16)) };
	l1 = { register_module("l1", torch::nn::Linear(16, 16)) };
	l2 = { register_module("l2", torch::nn::Linear(16, 8)) };
	l3 = { register_module("l3", torch::nn::Linear(8, action_size)) };
}

torch::Tensor policy::Continuous::forward(torch::Tensor x)
{
	x = torch::leaky_relu(l0->forward(x));
	x = torch::leaky_relu(l1->forward(x));
	x = torch::leaky_relu(l2->forward(x));
	x = torch::leaky_relu(l3->forward(x), 1);

	return x;
}

void policy::Continuous::act(const std::vector<float>& x, Environment& env, std::vector<Trajectory::Frame>& traj)
{
	auto x_t = torch::from_blob((void*)x.data(), {1, (long)x.size()}, torch::kFloat).clone();
	auto a_dist_params = forward(x_t);

	const auto k_speed = 0.1f;

	// a_dist_paras contains the mean and standard deviation of the normal distribution, use
	// those to sample a continuous action
	std::normal_distribution<float> c_dist(a_dist_params[0][0].item<float>(), a_dist_params[0][1].item<float>());
	std::normal_distribution<float> r_dist(a_dist_params[0][2].item<float>(), a_dist_params[0][3].item<float>());
	static std::default_random_engine gen;

	
	float u[2] = {r_dist(gen), c_dist(gen)};

	auto reward_t = env.step_reward(u);

	traj.push_back({x_t, a_dist_params, (unsigned)0, reward_t});
}



void policy::Continuous::train(const std::vector<Trajectory::Frame>& traj, float learning_rate)
{
	zero_grad();

	for (unsigned t = 0; t < traj.size(); t++)
	{
		const auto& f_t = traj[t];
		torch::log(f_t.action_probs * f_t.reward).backward();
	}

	for (auto& param : parameters())
	{
		param.data() += (param.grad() / static_cast<float>(traj.size())) * learning_rate;
	}
}