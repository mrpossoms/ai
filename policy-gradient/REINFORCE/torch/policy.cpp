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

	traj.push_back({x_t, a_probs, {}, (unsigned)a, reward_t});
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
	std::cout << "x:" << x_t << std::endl;
	assert(!torch::any(torch::isnan(x_t)).item<bool>());

	auto a_dist_params = forward(x_t);

	std::cout << "a_dist_params:" << a_dist_params << std::endl;

	// assert that the output is a 1x4 tensor and not nan
	assert(!torch::any(torch::isnan(a_dist_params)).item<bool>());
	assert(a_dist_params.sizes() == torch::IntArrayRef({1, 4}));



	const auto k_speed = 0.1f;

	// a_dist_paras contains the mean and standard deviation of the normal distribution, use
	// those to sample a continuous action
	
	auto mu_c = a_dist_params[0][0].item<float>();
	auto mu_r = a_dist_params[0][1].item<float>();
	std::normal_distribution<float> c_dist(mu_c, 0.1); //a_dist_params[0][1].item<float>());
	std::normal_distribution<float> r_dist(mu_r, 0.1); //a_dist_params[0][3].item<float>());
	static std::default_random_engine gen;
	
	float u[2] = {r_dist(gen), c_dist(gen)};

	std::cout << "u:" << u[0] << " " << u[1] << std::endl;


	auto reward_t = env.step_reward(u);

	traj.push_back({x_t, a_dist_params, torch::from_blob(u, {1, 2}, torch::kFloat).clone(), (unsigned)0, reward_t});
}


using namespace torch::indexing;

void policy::Continuous::train(const std::vector<Trajectory::Frame>& traj, float learning_rate)
{
	zero_grad();

	for (unsigned t = 0; t < traj.size(); t++)
	{
		const auto& f_t = traj[t];

		auto mu = f_t.action_probs.index({0, Slice(0, 2)});
		auto sigma_2 = torch::ones({1, 2}) * 0.01;
		//f_t.action_probs.index({0, Slice(2, 4)}).pow(2);

		// (torch::log(f_t.action_probs - f_t.action) * f_t.reward).backward();
		(((((mu - f_t.action).pow(2) / sigma_2) + torch::log(sigma_2 * 2 * M_PI)) * -0.5).sum()).backward();
	}

	for (auto& param : parameters())
	{
		param.data() += (param.grad() / static_cast<float>(traj.size())) * learning_rate;
	}
}