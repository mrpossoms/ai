#pragma once
#include <cereal/types/vector.hpp>

template<typename MatType=arma::mat>
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
		auto loss_sum = R(action_rewards.row(action_rewards.n_rows-1)); // * arma::accu(arma::log(prediction));
		return -loss_sum;
	}

	void Backward(const MatType& prediction,
	              const MatType& action_rewards,
	              MatType& gradient)
	{
		arma::mat action = action_rewards.submat(0, 0, action_rewards.n_rows-2, action_rewards.n_cols-1);
		gradient.set_size(size(action));

		for (unsigned c = 0; c < prediction.n_cols; c++)
		{
			gradient.col(c) = -(pow(m_gamma, c) * action_rewards.col(c)[action_rewards.n_rows-1]) / action.col(c);
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