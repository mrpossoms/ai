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

	// torch::autograd::AnomalyMode::set_enabled(true);

	// if (!mlpack::data::Load("model.json", "model", model))
	// {
	// 	std::cout << "Failed to load model." << std::endl;
	// }
	// else
	// {
	// 	LOADED = true;
	// }
}

bool net::loaded()
{
	return LOADED;
}

torch::Tensor net::policy_loss(const RL::Trajectory& traj)
{
	float R = 0;

	std::vector<float> returns;
	for (int i = 0; i <= traj.rewards.size(); i++)
	{
		R = traj.rewards[i] + 0.999f * R;
		returns.push_back(R);
	}

	torch::Tensor returns_tensor = torch::tensor(returns);
	// returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8);

	// std::cout << "RRRRRRRRRRRRRRRRRRRRRRR" << std::endl;
	// std::cout << returns_tensor << std::endl;
	// assert(returns_tensor.isnan().sum().item<int>() == 0);

	// std::vector<torch::Tensor> policy_loss;
	// torch::Tensor policy_loss = torch::zeros({traj.rewards.size(), 1, 4});

	std::vector<torch::Tensor> losses;

	for (int i = 0; i < traj.rewards.size(); i++)
	{
		auto& prob = traj.action_probs[i];
		// std::cout << prob << std::endl;

		// should be -log()
		// policy_loss.push_back(-torch::log(prob) * returns_tensor[i]);
		
		// policy_loss[i] = -torch::log(prob) * returns_tensor[i];
		losses.push_back(-torch::log(prob) * returns_tensor[i]);
	}

	// std::cout << "-------------------" << std::endl;
	// std::cout << policy_loss << std::endl;

	// torch::Tensor policy_loss_tensor = torch::stack(policy_loss);
	// auto policy_loss_mu = policy_loss_tensor.mean(); // / traj.rewards.size();
	// policy_loss_mu.requires_grad_(true);
	return torch::cat(losses).sum();



	// return policy_loss.sum();
}

void net::train_policy_gradient(const RL::Trajectory& traj, const net::hyper_parameters& hp)
{
	// model.train();


	torch::optim::Adam optimizer(
		model.parameters(),
		torch::optim::AdamOptions(0.01)
	);

	optimizer.zero_grad();


	// policy_loss_mu.requires_grad_(true);

	// auto policy_loss_mu = (-torch::log(traj.action_probs[0]) * returns_tensor[0]).mean();

	// std::cout << policy_loss_mu << std::endl;
	auto loss = policy_loss(traj);

	// assert policy_loss_tensor is not nan
	assert(!std::isnan(loss.template item<float>()));
	
	// copilot generated this, check later
	loss.backward();
	optimizer.step();

	// assert that model parameters are not nan
	for (auto& param : model.parameters())
	{
		// std::cout << "---------" << std::endl;
		// std::cout << param << std::endl;
		assert(param.isnan().sum().item<int>() == 0);
	}

	// model.eval();

	// for (int64_t epoch = 1; epoch <= hp.epochs; ++epoch) 
	// {
  	// 	int64_t batch_index = 0;

  	// 	model.zero_grad();

	// }
	// model.Train(
	// 	traj.state,
	// 	action_reward_trajectory,
	// 	optimizer);

	// if (!mlpack::data::Save("model.json", "model", model, true))
	// {
	// 	std::cout << "Failed to save model." << std::endl;
	// }
}

torch::Tensor net::act_probs(torch::Tensor x)
{
	auto probs = model.forward(x);

	// float epsilon
	auto eps = std::numeric_limits<float>::epsilon();

	probs = probs.clamp(eps, 1.f - eps);

	return probs;
}

RL::Action net::act(torch::Tensor probs)
{
	probs /= probs.sum();

	auto p = probs * 1000;

	// std::cout << p << std::endl;

	std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d({
    	p[0][0].item<float>(),
    	p[0][1].item<float>(),
    	p[0][2].item<float>(),
    	p[0][3].item<float>()
    });

	// model.Predict(_x, _a);

    auto u_idx = d(gen);
    assert(u_idx >= 0 && u_idx < 4);

    // std::cout << u_idx << std::endl;

    // chose one discrete action
    // TODO: reimplement this using continuous action space
	RL::Action a = {};
	a.u[u_idx] = 1;

	return a;
}
