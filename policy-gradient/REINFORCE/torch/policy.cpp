#include "policy.hpp"
#include <random>
// #include "policy_gradient.hpp"
// #define MLPACK_ENABLE_ANN_SERIALIZATION
// #include <mlpack/core.hpp>
// #include <mlpack/methods/ann.hpp>

// using namespace mlpack;


// static FFN<PolicyGradientLoss<arma::mat>, RandomInitialization, arma::mat> model;

// ens::Adam optimizer;

// #define DEBUG

policy::Discrete::Discrete()
{
	const auto observation_size = 4;
	const auto action_size = 4;

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
	auto action = torch::ones({1, 4}) * 0.01;
	action.index_put_({0, a}, 1);
	traj.push_back({x_t, a_probs, action, (unsigned)a, reward_t});
}



void policy::Discrete::train(const std::vector<Trajectory::Frame>& traj, float learning_rate)
{
	zero_grad();

	for (unsigned t = 0; t < traj.size(); t++)
	{
		const auto& f_t = traj[t];
		(torch::log(f_t.action_probs.flatten()[f_t.action_idx]) * f_t.reward).backward();
	}

	for (auto& param : parameters())
	{
		param.data() += (param.grad() / static_cast<float>(traj.size())) * learning_rate;
	}
}


policy::Continuous::Continuous()
{
	const auto observation_size = 4;
	const auto action_size = 2;

	l0 = { register_module("l0", torch::nn::Linear(observation_size, 16)) };
	l1 = { register_module("l1", torch::nn::Linear(16, 16)) };
	l2 = { register_module("l2", torch::nn::Linear(16, action_size)) };
}

torch::Tensor policy::Continuous::forward(torch::Tensor x)
{
	x = torch::leaky_relu(l0->forward(x));
	x = torch::leaky_relu(l1->forward(x));
	x = torch::leaky_relu(l2->forward(x), 1);

	return x;
}

void policy::Continuous::act(const std::vector<float>& x, Environment& env, std::vector<Trajectory::Frame>& traj)
{
	auto x_t = torch::from_blob((void*)x.data(), {1, (long)x.size()}, torch::kFloat).clone();
	assert(!torch::any(torch::isnan(x_t)).item<bool>());

	auto a_dist_params = forward(x_t);
#ifdef DEBUG
	std::cout << "x:" << x_t << std::endl;
	std::cout << "a_dist_params:" << a_dist_params << std::endl;
#endif
	// assert that the output is a 1x4 tensor and not nan
	assert(!torch::any(torch::isnan(a_dist_params)).item<bool>());
	assert(a_dist_params.sizes() == torch::IntArrayRef({1, 4}));



	const auto k_speed = 0.1f;

	// a_dist_paras contains the mean and standard deviation of the normal distribution, use
	// those to sample a continuous action
	

	auto mu = a_dist_params.index({0, Slice(0, 2)});
	// auto sigma_2 = torch::ones({1, 2}) * 1;
	auto sigma = torch::log(torch::exp(a_dist_params.index({0, Slice(2, 4)})) + 1);
	std::normal_distribution<float> c_dist(mu[0].item<float>(), sigma[0].item<float>());
	std::normal_distribution<float> r_dist(mu[1].item<float>(), sigma[1].item<float>());
	// std::normal_distribution<float> c_dist(mu[0].item<float>(), sigma);
	// std::normal_distribution<float> r_dist(mu[1].item<float>(), sigma);
	static std::default_random_engine gen;
	
	float u[2] = {r_dist(gen), c_dist(gen)};
#ifdef DEBUG
	std::cout << "u:" << u[0] << " " << u[1] << std::endl;
#endif

	auto reward_t = env.step_reward(u);

	traj.push_back({x_t, a_dist_params, torch::from_blob(u, {1, 2}, torch::kFloat).clone(), (unsigned)0, reward_t});
}


void policy::Continuous::train(const std::vector<Trajectory::Frame>& traj, float learning_rate)
{
	zero_grad();
	const auto sqrt_2pi = std::sqrt(2 * M_PI);

	for (unsigned t = 0; t < traj.size(); t++)
	{
		const auto& f_t = traj[t];

		auto mu = f_t.action_probs.index({0, Slice(0, 2)});
		auto sigma = torch::log(torch::exp(f_t.action_probs.index({0, Slice(2, 4)})) + 1);

		auto probs = ((1/(sigma * sqrt_2pi)) * torch::exp(-0.5 * ((f_t.action - mu) / sigma).pow(2)));
		(probs.prod() * f_t.reward).backward();
	}

	for (auto& param : parameters())
	{
		auto g = param.grad() / static_cast<float>(traj.size());
		param.data() += g * learning_rate;
	}
}