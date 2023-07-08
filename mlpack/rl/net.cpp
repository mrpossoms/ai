#include "net.hpp"
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>

using namespace mlpack;

template<size_t action_size, typename MatType=arma::mat>
struct PolicyGradientLoss
{
	PolicyGradientLoss(float gamma = 0.999f) : m_gamma(gamma) {}

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
		return -loss_sum;
	}

	void Backward(const MatType& prediction,
	              const MatType& action_rewards,
	              MatType& gradient)
	{
		gradient.set_size(size(prediction));

		for (unsigned c = 0; c < prediction.n_cols; c++)
		{
			gradient.col(c) = -(pow(m_gamma, c) * action_rewards.col(c)[action_size]) / prediction.col(c);
		}
	}

	template<typename Archive>
	void serialize(
	    Archive& ar,
	    const uint32_t /* version */)
	{
	    ar(CEREAL_NVP(m_gamma));
	}

private:
	float m_gamma;
};

static FFN<PolicyGradientLoss<2, arma::mat>, RandomInitialization, arma::mat> model;

ens::Adam optimizer;
bool LOADED = false;

void net::init(size_t observation_size, size_t action_size)
{
	model.Add<LinearType<arma::mat, NoRegularizer>>(observation_size);
	model.Add<LinearType<arma::mat, NoRegularizer>>(16);
	// model.Add<LeakyReLUType<arma::mat>>();
	model.Add<LinearType<arma::mat, NoRegularizer>>(4);
	// model.Add<LeakyReLUType<arma::mat>>();
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
	arma::mat _a(2,1);
	_x.row(0)[0] = x.d_goal[0];
	_x.row(1)[0] = x.d_goal[1];
	_x.row(2)[0] = x.vel[0];
	_x.row(3)[0] = x.vel[1];

	model.Predict(_x, _a);

	return { _a.row(0)[0], _a.row(1)[0], };
}