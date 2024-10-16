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
	traj.push_back({x_t, a_probs, a_probs, action, (unsigned)a, reward_t});
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
	l0 = { register_module("l0", torch::nn::Linear(observation_size(), output_size())) };
	
	//l0 = { register_module("l0", torch::nn::Linear(observation_size(), 16)) };
	//l1 = { register_module("l1", torch::nn::Linear(16, 16)) };
	//l2 = { register_module("l2", torch::nn::Linear(16, output_size())) };
}

torch::Tensor policy::Continuous::forward(torch::Tensor x)
{
	x = l0->forward(x);
	//x = torch::leaky_relu(l0->forward(x));
	//x = torch::leaky_relu(l1->forward(x));
	//x = torch::leaky_relu(l2->forward(x), 1);

	return x;
}

torch::Tensor policy::Continuous::tensor_from_state(Environment& env)
{
	const auto x = env.get_state_vector();
	auto x_t = torch::from_blob((void*)x.data(), {1, (long)x.size()}, torch::kFloat).clone();
	assert(!torch::any(torch::isnan(x_t)).item<bool>());
	return x_t;
}

torch::Tensor policy::Continuous::action_sigma(const torch::Tensor& a_dist_params)
{
	//return torch::ones({2}) * 0.1f;
	return torch::log(torch::exp(a_dist_params.index({0, Slice(action_size(), output_size())})) + 1);// + 0.01f;
}

torch::Tensor policy::gaussian(const torch::Tensor& x, const torch::Tensor& mu, const torch::Tensor& var)
{
	// auto mag = 1/(torch::sqrt(var * 2 * M_PI));
	auto g =  torch::exp(-((x - mu).pow(2) / (2 * var)));
	return g;// / mag;
	// auto exp = 1 / torch::exp(((x - mu).pow(2) / (var)));
	// return exp * (1 - exp) * 0.05 * torch::sqrt((x-mu).pow(2));
}

torch::Tensor policy::Continuous::action_probabilities(const torch::Tensor& a_dist_params, const torch::Tensor& a)
{
	constexpr auto sqrt_2pi = std::sqrt(2 * M_PI);
	auto mu = a_dist_params.index({0, Slice(0, action_size())});
	auto sigma = action_sigma(a_dist_params);
	auto var = sigma.pow(2);

	// auto eps = 1e-3;
	// auto a_eps = torch::ones_like(a) * eps;
	// return torch::abs(gaussian(a + a_eps, mu, var) - gaussian(a, mu, var)) * eps;

	// return torch::clamp(gaussian(a, mu, var), 0.0001f, 0.9999f);

	// return ((1/(sigma * sqrt_2pi)) * torch::exp(-0.5 * ((f_t.action - mu) / sigma).pow(2)));
	return ((1/(sigma * sqrt_2pi)) * torch::exp(-0.5 * ((a - mu) / sigma).pow(2)));
}

const torch::Tensor policy::Continuous::act(Environment& env, trajectory::Trajectory& traj)
{
	auto x_t = policy::Continuous::tensor_from_state(env);
	auto output = forward(x_t);

// #ifdef DEBUG
	// std::cout << "x:" << x_t << std::endl;
	// std::cout << "output:" << output << std::endl;
// #endif
	// assert that the output is a 1x4 tensor and not nan
	if (torch::any(torch::isnan(output)).item<bool>())
	{
		std::cout << "x_t: " << x_t << std::endl;
		std::cout << "output: " << output << std::endl;
		std::cout << "-------------------------------" << std::endl;
		print_params();
		assert(!torch::any(torch::isnan(output)).item<bool>());
	}

	assert(output.sizes() == torch::IntArrayRef({1, 4}));

	auto mu = output.index({0, Slice(0, action_size())});
	auto sigma = action_sigma(output);
	std::normal_distribution<float> c_dist(mu[1].item<float>(), sigma[1].item<float>());
	std::normal_distribution<float> r_dist(mu[0].item<float>(), sigma[0].item<float>());

	static std::default_random_engine gen;
	
	float u[2] = {r_dist(gen), c_dist(gen)};
#ifdef DEBUG
	std::cout << "u:" << u[0] << " " << u[1] << std::endl;
#endif

	auto reward = env.step_reward(u);
	auto reward_t = torch::from_blob(&reward, {1}, torch::kFloat).clone();
	auto action_t = torch::from_blob(u, {1, 2}, torch::kFloat).clone();
	auto a_probs = action_probabilities(output, action_t);

	traj.push_back({x_t, output, a_probs, action_t, (unsigned)0, reward_t});

	return a_probs;
}

void policy::Continuous::train(const trajectory::Trajectory& traj, Policy& policy, float learning_rate)
{
	policy.zero_grad();

/////////////////////////////////
	// std::cout << "traj.action_probs: " << traj.action_probs << std::endl;
	auto prob_prods = traj.action_probs.prod(1);
	// std::cout << "prob_prods: "<< prob_prods << std::endl;
	auto r_sum = (prob_prods * traj.rewards).sum();
	// std::cout << "r_sum: "<< r_sum << std::endl;
	auto r = r_sum / static_cast<float>(traj.size());
	// std::cout << "r:" << r << std::endl;

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

////////////////////////////////

	for (auto& param : policy.parameters())
	{
		auto g = param.grad();
		std::cout << "g: " << g << std::endl;
		param.data() += g * learning_rate;
	}
}

static torch::Tensor action_probabilities(const torch::Tensor& a, const torch::Tensor& mu, const torch::Tensor& var)
{
	auto prob = policy::gaussian(a, mu, var);
	if (torch::any(prob <= 0).item<bool>())
	{
		std::cout << "a: " << a << std::endl;
		std::cout << "mu: " << mu << std::endl;
		std::cout << "var: " << var << std::endl;
		std::cout << "prob: " << prob << std::endl;
		assert(false);
	}
	
	return prob;
}

void policy::Continuous::train(const trajectory::Trajectory& traj, float learning_rate)
{
	zero_grad();

// 	constexpr auto sqrt_2pi = std::sqrt(2 * M_PI);
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
// 		auto sigma = torch::ones({2}) * 0.3f; //torch::log(torch::exp(f_t.action_probs.index({Slice(2, 4)})) + 1);
// #ifdef DEBUG
// 		std::cout << "sigma: " << sigma << std::endl;
// #endif
// 		auto probs = ((1/(sigma * sqrt_2pi)) * torch::exp(-0.5 * ((f_t.action.index({Slice(0, 2)}) - mu) / sigma).pow(2)));
// 		(probs.prod() * f_t.reward).backward();
// 	}

	// auto mus = traj.action_probs.index({Slice(0, traj.size()), Slice(0, 2)});

	// auto u = traj.actions.index({Slice(0, traj.size()), Slice(0, 2)});
	// auto probs = ((1/(sigmas * sqrt_2pi)) * torch::exp(-0.5 * ((u - mus) / sigmas).pow(2)));


/////////////////////////////////
	// std::cout << "traj.action_probs: " << traj.action_probs << std::endl;
	auto prob_prods = traj.action_probs.prod(1);
	// std::cout << "prob_prods: "<< prob_prods << std::endl;
	auto r_sum = (prob_prods * traj.rewards).sum();
	// std::cout << "r_sum: "<< r_sum << std::endl;
	auto r = r_sum / static_cast<float>(traj.size());
	// std::cout << "r:" << r << std::endl;

	r.backward(); //{}, retain_graph={true});

////////////////////////////////

	for (auto& param : parameters())
	{
		auto g = param.grad();

		if (torch::any(torch::isnan(g)).item<bool>() || torch::all(g == 0).item<bool>())
		{
			std::cout << "g: " << g << std::endl;
			std::cout << "--------------------------------\n";
			print_params();
			std::cout << "--------------------------------\n";
			std::cout << "states: " << traj.states << std::endl;
			std::cout << "outputs: " << traj.outputs << std::endl;
			std::cout << "actions:" << traj.actions << std::endl;
			std::cout << "action_probs: " << traj.action_probs << std::endl;
			std::cout << "prob_prods: " << prob_prods << std::endl;
			std::cout << "rewards: " << traj.rewards << std::endl;
			std::cout << "r: " << r << std::endl;		

			assert(false);
		}

		param.data() += g * learning_rate;
	}
}
