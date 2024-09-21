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
	l0 = { register_module("l0", torch::nn::Linear(observation_size(), 16)) };
	l1 = { register_module("l1", torch::nn::Linear(16, 16)) };
	l2 = { register_module("l2", torch::nn::Linear(16, 8)) };
	l3 = { register_module("l3", torch::nn::Linear(8, output_size())) };
}

torch::Tensor policy::Discrete::forward(torch::Tensor x)
{
	x = torch::leaky_relu(l0->forward(x));
	x = torch::leaky_relu(l1->forward(x));
	x = torch::leaky_relu(l2->forward(x));
	x = torch::softmax(l3->forward(x), 1);

	return x;
}

const torch::Tensor policy::Discrete::act(Environment& env, trajectory::Trajectory& traj)
{
	const auto x = env.get_state_vector();
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

	// create tensor from scalar reward
	auto reward = env.step_reward(u);
	auto reward_t = torch::from_blob(&reward, {1}, torch::kFloat).clone();
	auto action = torch::ones({1, 4}) * 0.01;
	action.index_put_({0, a}, 1);
	traj.push_back({x_t, a_probs, action, (unsigned)a, reward_t});
	return a_probs;
}



void policy::Discrete::train(const trajectory::Trajectory& traj, float learning_rate)
{
	zero_grad();

	for (unsigned t = 0; t < traj.size(); t++)
	{
		const auto f_t = traj[t];
		(torch::log(f_t.action_probs.flatten()[f_t.action_idx]) * f_t.reward).backward();
	}

	for (auto& param : parameters())
	{
		param.data() += (param.grad() / static_cast<float>(traj.size())) * learning_rate;
	}
}


policy::Continuous::Continuous()
{
	l0 = { register_module("l0", torch::nn::Linear(observation_size(), 16)) };
	l1 = { register_module("l1", torch::nn::Linear(16, 16)) };
	l2 = { register_module("l2", torch::nn::Linear(16, output_size())) };
}

torch::Tensor policy::Continuous::forward(torch::Tensor x)
{
	x = torch::leaky_relu(l0->forward(x));
	x = torch::leaky_relu(l1->forward(x));
	x = torch::leaky_relu(l2->forward(x), 1);

	return x;
}

const torch::Tensor policy::Continuous::act(Environment& env, trajectory::Trajectory& traj)
{
	const auto x = env.get_state_vector();
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

	auto mu = a_dist_params.index({0, Slice(0, action_size())});
	auto sigma = torch::log(torch::exp(a_dist_params.index({0, Slice(action_size(), output_size())})) + 1) + 0.01f;
	std::normal_distribution<float> c_dist(mu[0].item<float>(), sigma[0].item<float>());
	std::normal_distribution<float> r_dist(mu[1].item<float>(), sigma[1].item<float>());

	static std::default_random_engine gen;
	
	float u[2] = {r_dist(gen), c_dist(gen)};
#ifdef DEBUG
	std::cout << "u:" << u[0] << " " << u[1] << std::endl;
#endif

	auto reward = env.step_reward(u);
	auto reward_t = torch::from_blob(&reward, {1}, torch::kFloat).clone();

	constexpr auto sqrt_2pi = std::sqrt(2 * M_PI);
	auto u_t = torch::from_blob(u, {1, 2}, torch::kFloat).clone();
	auto a_probs = ((1/(sigma * sqrt_2pi)) * torch::exp(-0.5 * ((u_t - mu) / sigma).pow(2)));

	traj.push_back({x_t, a_probs, u_t, (unsigned)0, reward_t});

	return a_probs;
}


void policy::Continuous::train(const trajectory::Trajectory& traj, float learning_rate)
{
	zero_grad();

// 	for (unsigned t = 0; t < traj.size(); t++)
// 	{
// 		const auto f_t = traj[t];
// #ifdef DEBUG
// 		std::cout << "f_t.action_probs: " << f_t.action_probs << std::endl;
// 		std::cout << "f_t.action: " << f_t.action << std::endl;
// 		std::cout << "f_t.reward: " << f_t.reward << std::endl;
// #endif

// 		auto mu = f_t.action_probs.index({Slice(0, 2)});
// #ifdef DEBUG
// 		std::cout << "mu: " << mu << std::endl;
// #endif
// 		auto sigma = torch::log(torch::exp(f_t.action_probs.index({Slice(2, 4)})) + 1);
// #ifdef DEBUG
// 		std::cout << "sigma: " << sigma << std::endl;
// #endif
// 		auto probs = ((1/(sigma * sqrt_2pi)) * torch::exp(-0.5 * ((f_t.action.index({Slice(0, 2)}) - mu) / sigma).pow(2)));
// 		(probs.prod() * f_t.reward).backward();
// 	}

	// auto mus = traj.action_probs.index({Slice(0, traj.size()), Slice(0, 2)});

	// auto u = traj.actions.index({Slice(0, traj.size()), Slice(0, 2)});
	// auto probs = ((1/(sigmas * sqrt_2pi)) * torch::exp(-0.5 * ((u - mus) / sigmas).pow(2)));
	auto prob_prods = traj.action_probs.prod(1);
	auto r = (prob_prods * traj.rewards).sum() / static_cast<float>(traj.size());

#ifdef DEBUG
	std::cout << "action_probs: " << traj.action_probs << std::endl;
	std::cout << "us: " << u << std::endl;
	std::cout << "mus: " << mus << std::endl;
	std::cout << "sigmas: " << sigmas << std::endl;
	std::cout << "probs: " << probs << std::endl;
	std::cout << "prob_prods: " << prob_prods << std::endl;
	std::cout << "rewards: " << traj.rewards << std::endl;
	std::cout << "r: " << r << std::endl;
	exit(0);
#endif

	r.backward(); //{}, retain_graph={true});

// 	for (unsigned t = 0; t < traj.size(); t++)
// 	{
// 		const auto f_t = traj[t];
// #ifdef DEBUG
// 		std::cout << "f_t.action_probs: " << f_t.action_probs << std::endl;
// 		std::cout << "f_t.action: " << f_t.action << std::endl;
// 		std::cout << "f_t.reward: " << f_t.reward << std::endl;
// #endif

// 		auto mu = f_t.action_probs.index({Slice(0, 2)});
// #ifdef DEBUG
// 		std::cout << "mu: " << mu << std::endl;
// #endif
// 		auto sigma = torch::log(torch::exp(f_t.action_probs.index({Slice(2, 4)})) + 1);
// #ifdef DEBUG
// 		std::cout << "sigma: " << sigma << std::endl;
// #endif
// 		auto probs = ((1/(sigma * sqrt_2pi)) * torch::exp(-0.5 * ((f_t.action.index({Slice(0, 2)}) - mu) / sigma).pow(2)));
// 		(probs.prod() * f_t.reward).backward({}, true);
// 	}

	for (auto& param : parameters())
	{
		auto g = param.grad();
		param.data() += g * learning_rate;
	}
}