#include "net.hpp"
#include <mlpack.hpp>

using namespace mlpack;

template<size_t action_size, typename MatType=arma::mat>
struct PolicyGradientLoss
{
	PolicyGradientLoss(float gamma = 0.99f) : m_gamma(gamma) {}

	typename
	MatType::elem_type R(const MatType& rewards)
	{
		float r = {};
		for (unsigned i = 0; i < rewards.n_cols; i++)
		{
			r += pow(m_gamma, i) * rewards.col(i)[0];
		}
		
		return r;
	}

	typename
	MatType::elem_type Forward(const MatType& prediction,
	                                      const MatType& action_rewards)
	{
		auto loss_sum = R(action_rewards.row(action_size)); // * arma::accu(arma::log(prediction));
		return loss_sum;
	}

	void Backward(const MatType& prediction,
	              const MatType& action_rewards,
	              MatType& gradient)
	{
		gradient.set_size(size(prediction));

		// std::cout << "pred: " << prediction << std::endl << "action: " << action_rewards << std::endl;

		// for (unsigned c = 0; c < prediction.n_cols; c++)
		// {
		// 	gradient.col(c) = arma::log(prediction.col(c)) * pow(m_gamma, c) * action_rewards.col(c)[action_size];
		// }

		for (unsigned c = 0; c < prediction.n_cols; c++)
		{
			gradient.col(c) = -(pow(m_gamma, c) * action_rewards.col(c)[action_size]) / prediction.col(c);
		}

		// std::cout << gradient << std::endl;
		// sleep(1);

		// gradient = -R(action_rewards.row(action_size-1)) / prediction;
		// gradient = arma::log(prediction) * R(action_rewards.row(action_size-1));
	}


private:
	float m_gamma;
};

static FFN<PolicyGradientLoss<2, arma::fmat>, RandomInitialization, arma::fmat> model;

void net::init(size_t observation_size, size_t action_size)
{
	model.Add<LinearType<arma::fmat>>(observation_size);
	model.Add<LinearType<arma::fmat>>(8);
	model.Add<LinearType<arma::fmat>>(action_size);
}

void net::train_policy_gradient(const RL::Trajectory& trajectory, const net::hyper_parameters& hp)
{
	ens::Adam optimizer(
		hp.learning_rate,  // Step size of the optimizer.
		trajectory.states.n_cols, // Batch size. Number of data points that are used in each
		            // iteration.
		0.9,        // Exponential decay rate for the first moment estimates.
		0.999, // Exponential decay rate for the weighted infinity norm estimates.
		1e-8,  // Value used to initialise the mean squared gradient parameter.
		hp.epochs, // Max number of iterations.
		1e-8,           // Tolerance.
		false);

	model.Train(
		trajectory.states,
		trajectory.action_rewards,
		optimizer);
}

RL::Action net::act(const RL::State& x)
{
	arma::fmat _x(4,1);
	arma::fmat _a(2,1);
	_x.row(0)[0] = x.d_goal[0];
	_x.row(1)[0] = x.d_goal[1];
	_x.row(2)[0] = x.vel[0];
	_x.row(3)[0] = x.vel[1];

	model.Predict(_x, _a);

	return { _a.row(0)[0], _a.row(1)[0], };
}