#include "policy.hpp"
#include <random>
// #include "policy_gradient.hpp"
// #define MLPACK_ENABLE_ANN_SERIALIZATION
// #include <mlpack/core.hpp>
// #include <mlpack/methods/ann.hpp>

// using namespace mlpack;


// static FFN<PolicyGradientLoss<arma::mat>, RandomInitialization, arma::mat> model;

// ens::Adam optimizer;
bool LOADED = false;

struct Net : torch::nn::Module
{
	Net() = default;

	Net(unsigned obs_size, unsigned act_size)
	{
		l0 = { register_module("l0", torch::nn::Linear(obs_size, 16)) };
		l1 = { register_module("l1", torch::nn::Linear(16, 16)) };
		l2 = { register_module("l2", torch::nn::Linear(16, 8)) };
		l3 = { register_module("l3", torch::nn::Linear(8, act_size)) };
	}

	// Net(const )

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::leaky_relu(l0->forward(x));
		x = torch::leaky_relu(l1->forward(x));
		x = torch::leaky_relu(l2->forward(x));
		x = torch::softmax(l3->forward(x), 1);

		return x;
	}

	torch::nn::Linear l0 = nullptr, l1 = nullptr, l2 = nullptr, l3 = nullptr;
};

std::shared_ptr<Net> model;
std::unique_ptr<torch::optim::Adam> optimizer;

void policy::init(size_t observation_size, size_t action_size)
{
	model = std::make_shared<Net>(observation_size, action_size);

	// try to load the model
	try
	{
		torch::load(model, std::string("model.pt"));
		LOADED = true;
	}
	catch (const c10::Error& e)
	{
		LOADED = false;

	}

	optimizer = std::make_unique<torch::optim::Adam>(
		model->parameters(),
		torch::optim::AdamOptions(0.001)
	);
}

void policy::save(const std::string& path)
{
	torch::save(model, path);
}

bool policy::loaded()
{
	return LOADED;
}


void policy::train_policy_gradient(const std::vector<Trajectory::Frame>& traj, const policy::hyper_parameters& hp)
{
	model->zero_grad();

	for (unsigned t = 0; t < traj.size(); t++)
	{
		const auto& f_t = traj[t];
		// std::cout << f_t.action_probs << std::endl;
		(torch::log(torch::flatten(f_t.action_probs)[f_t.action_idx]) * f_t.reward).backward();
	}

	for (auto& param : model->parameters())
	{
		param.data() += (param.grad() / static_cast<float>(traj.size())) * 0.01f;
	}


}

torch::Tensor policy::act_probs(torch::Tensor x)
{
	auto probs = model->forward(x);

	// float epsilon
	// auto eps = std::numeric_limits<float>::epsilon();

	// probs = probs.clamp(eps, 1.f - eps);

	return probs;
}

int policy::act(torch::Tensor probs, bool stochastic)
{
	int a = 0;
	if (stochastic)
	{
		auto index = torch::multinomial(probs, 1);
		a = index.item<int>();
	}
	else
	{
		a =  torch::argmax(probs, 1).item<int>();
	}

	// if (a >= 2)
	// {
	// 	std::cout << "action: " << a << std::endl;
	// }

	return a;
}

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