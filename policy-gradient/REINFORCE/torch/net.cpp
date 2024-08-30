#include "net.hpp"
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
		layers[0] = { register_module("l0", torch::nn::Linear(obs_size, 16)) };
		layers[1] = { register_module("l1", torch::nn::Linear(16, 4)) };
		layers[2] = { register_module("l2", torch::nn::Linear(4, act_size)) };
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::leaky_relu(layers[0]->forward(x));
		x = torch::leaky_relu(layers[1]->forward(x));
		x = torch::softmax(layers[2]->forward(x), 1);

		return x;
	}

	torch::nn::Linear layers[3] = {
		nullptr,
		nullptr,
		nullptr,
	};
};

Net model;
std::unique_ptr<torch::optim::Adam> optimizer;

void net::init(size_t observation_size, size_t action_size)
{
	model = Net(observation_size, action_size);

	optimizer = std::make_unique<torch::optim::Adam>(
		model.parameters(),
		torch::optim::AdamOptions(0.001)
	);
}

bool net::loaded()
{
	return LOADED;
}


void net::train_policy_gradient(const std::vector<RL::Trajectory::Frame>& traj, const net::hyper_parameters& hp)
{
	model.zero_grad();

	for (unsigned t = 0; t < traj.size(); t++)
	{
		const auto& f_t = traj[t];
		// std::cout << f_t.action_probs << std::endl;
		(torch::log(torch::flatten(f_t.action_probs)[f_t.action_idx]) * f_t.reward).backward();
	}

	for (auto& param : model.parameters())
	{
		param.data() += (param.grad() / static_cast<float>(traj.size())) * 0.01f;
	}


}

torch::Tensor net::act_probs(torch::Tensor x)
{
	auto probs = model.forward(x);

	// float epsilon
	auto eps = std::numeric_limits<float>::epsilon();

	probs = probs.clamp(eps, 1.f - eps);

	return probs;
}

int net::act(torch::Tensor probs, bool stochastic)
{
	if (stochastic)
	{
		auto index = torch::multinomial(probs, 1);
		return index.item<int>();
	}

	return torch::argmax(probs, 1).item<int>();
}
