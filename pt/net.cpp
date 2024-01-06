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
		for (unsigned i = 0; i < sizeof(layers) / sizeof(torch::nn::Linear); i++)
		{
			x = layers[i](x);
		}

		return x;
	}

	torch::nn::Linear layers[3] = {
		nullptr,
		nullptr,
		nullptr,
	};
};

Net model;

void net::init(size_t observation_size, size_t action_size)
{
	// model.Add<LinearType<arma::mat, NoRegularizer>>(observation_size);
	// model.Add<LinearType<arma::mat, NoRegularizer>>(16);
	// model.Add<LinearType<arma::mat, NoRegularizer>>(4);
	// model.Add<LinearType<arma::mat, NoRegularizer>>(action_size);

	model = Net(observation_size, action_size);

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

void net::train_policy_gradient(const RL::Trajectory& traj, const net::hyper_parameters& hp)
{
	static bool first = true;
	if (first)
	{
		// optimizer = ens::Adam(
		// 	hp.learning_rate,  // Step size of the optimizer.
		// 	traj.state.n_cols, // Batch size. Number of data points that are used in each
		// 	            // iteration.
		// 	0.9,        // Exponential decay rate for the first moment estimates.
		// 	0.999, // Exponential decay rate for the weighted infinity norm estimates.
		// 	1e-8,  // Value used to initialise the mean squared gradient parameter.
		// 	hp.epochs, // Max number of iterations.
		// 	1e-8,           // Tolerance.
		// 	false);
		// first = false;
	}

	torch::optim::Adam optimizer(
    model.parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

	float R = 0;
	torch::Tensor returns = torch::zeros({traj.rewards.size(), 1});
	
	for (int i = traj.rewards.size() - 1; i >= 0; i--)
	{
		R = traj.rewards[i] + 0.999f * R;
		returns[i] = R;
	}

	returns = (returns - returns.mean()) / (returns.std() + 1e-8);

	torch::Tensor policy_loss = torch::zeros({traj.rewards.size(), 4});
	for (int i = 0; i < traj.rewards.size(); i++)
	{
		auto& prob = traj.action_probs[i];
		policy_loss[i] = -torch::log(prob) * returns[i];
	}

	optimizer.zero_grad();

	// copilot generated this, check later
	policy_loss.mean().backward();
	optimizer.step();


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

torch::Tensor net::act_probs(const RL::State& x)
{
	torch::Tensor _x = torch::zeros({1, 4});

	_x[0][0]= x.d_goal[0];
	_x[0][1]= x.d_goal[1];
	_x[0][2]= x.vel[0];
	_x[0][3]= x.vel[1];

	return model.forward(_x);
}

RL::Action net::act(torch::Tensor& probs)
{
	auto p = probs * 1000;

	std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d({
    	p[0][0].item<int>(),
    	p[0][1].item<int>(),
    	p[0][2].item<int>(),
    	p[0][3].item<int>()
    });

	// model.Predict(_x, _a);

    // chose one discrete action
    // TODO: reimplement this using continuous action space
	RL::Action a;
	a.u[d(gen)] = 1;

	return a;
}