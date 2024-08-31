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

void net::init(size_t observation_size, size_t action_size)
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

void net::save(const std::string& path)
{
	torch::save(model, path);
}

bool net::loaded()
{
	return LOADED;
}


void net::train_policy_gradient(const std::vector<Trajectory::Frame>& traj, const net::hyper_parameters& hp)
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

torch::Tensor net::act_probs(torch::Tensor x)
{
	auto probs = model->forward(x);

	// float epsilon
	// auto eps = std::numeric_limits<float>::epsilon();

	// probs = probs.clamp(eps, 1.f - eps);

	return probs;
}

int net::act(torch::Tensor probs, bool stochastic)
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
