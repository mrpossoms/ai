#include "net.hpp"
#include <mlpack.hpp>

using namespace mlpack;

static FFN<HuberLoss, RandomInitialization, arma::fmat> model;

void net::init(size_t observation_size, size_t action_size)
{
	model.Add<LinearType<arma::fmat>>(observation_size);
	model.Add<SigmoidType<arma::fmat>>();
	model.Add<LinearType<arma::fmat>>(128);
	model.Add<SigmoidType<arma::fmat>>();
	model.Add<LinearType<arma::fmat>>(action_size);
}

void net::train_policy_gradient(const std::vector<std::tuple<state, action, float>>& trajectory, net::hyper_parameters& hp)
{
	ens::Adam optimizer(
		hp.learning_rate,  // Step size of the optimizer.
		hp.batch_size, // Batch size. Number of data points that are used in each
		            // iteration.
		0.9,        // Exponential decay rate for the first moment estimates.
		0.999, // Exponential decay rate for the weighted infinity norm estimates.
		1e-8,  // Value used to initialise the mean squared gradient parameter.
		hp.epochs, // Max number of iterations.
		1e-8,           // Tolerance.
		true);


}

action net::act(const state& x)
{
	arma::fmat _x(4,1);
	arma::fmat _a(2,1);
	_x.row(0)[0] = x.d_goal[0];
	_x.row(1)[0] = x.d_goal[1];
	_x.row(2)[0] = x.heading[0];
	_x.row(3)[0] = x.heading[1];

	model.Predict(_x, _a);

	return { _a.row(0)[0], _a.row(1)[0], };
}