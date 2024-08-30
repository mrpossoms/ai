#include "net.hpp"
#include "policy_gradient.hpp"
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>

using namespace mlpack;


static FFN<PolicyGradientLoss<arma::mat>, RandomInitialization, arma::mat> model;

ens::Adam optimizer;
bool LOADED = false;

void net::init(size_t observation_size, size_t action_size)
{
	model.Add<LinearType<arma::mat, NoRegularizer>>(observation_size);
	model.Add<LinearType<arma::mat, NoRegularizer>>(16);
	model.Add<LinearType<arma::mat, NoRegularizer>>(4);
	model.Add<LinearType<arma::mat, NoRegularizer>>(action_size);

	if (!mlpack::data::Load("model.json", "model", model))
	{
		std::cout << "Failed to load model." << std::endl;
	}
	else
	{
		LOADED = true;
	}
}

bool net::loaded()
{
	return LOADED;
}


void net::train_policy_gradient(const RL::Trajectory& trajectory, const net::hyper_parameters& hp)
{
	static bool first = true;
	if (first)
	{
		optimizer = ens::Adam(
			hp.learning_rate,  // Step size of the optimizer.
			trajectory.states.n_cols, // Batch size. Number of data points that are used in each
			            // iteration.
			0.9,        // Exponential decay rate for the first moment estimates.
			0.999, // Exponential decay rate for the weighted infinity norm estimates.
			1e-8,  // Value used to initialise the mean squared gradient parameter.
			hp.epochs, // Max number of iterations.
			1e-8,           // Tolerance.
			false);
		first = false;
	}

	model.Train(
		trajectory.states,
		trajectory.action_rewards,
		optimizer);

	if (!mlpack::data::Save("model.json", "model", model, true))
	{
		std::cout << "Failed to save model." << std::endl;
	}
}

RL::Action net::act(const RL::State& x)
{
	arma::mat _x(4,1);
	arma::mat _a(sizeof(RL::Action) / sizeof(float),1);
	_x.row(0)[0] = x.d_goal[0];
	_x.row(1)[0] = x.d_goal[1];
	_x.row(2)[0] = x.vel[0];
	_x.row(3)[0] = x.vel[1];

	model.Predict(_x, _a);

	RL::Action a;

	for (unsigned i = 0; i < sizeof(RL::Action) / sizeof(float); i++)
	{
		a.u[i] = _a(i, 0);
	}

	return a;
}