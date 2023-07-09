#include <armadillo>
#include <cassert>
#include "policy_gradient.hpp"

arma::mat cart_track_step(const arma::subview_col<double>& x_t)
{
	arma::mat stm({
		{1, 0.1},
		{0, 0.99},
	});
	return stm * x_t;
}

double R_t(const arma::subview_col<double>& x_t, const arma::subview_col<double>& x_t1)
{
	return pow(x_t[0], 2.0) - pow(x_t1[0], 2.0);
}

int main(int argc, char const *argv[])
{
	PolicyGradientLoss pg;

	arma::mat X(2, 10, arma::fill::randn);
	arma::mat A_R(2, 10);
	arma::mat W(2, 1, arma::fill::randn);

	for (unsigned t = 1; t < X.n_cols; t++)
	{
		auto u = (X.col(t-1) * W); // compute action
		A_R(0, t) = u;
		X(1, t-1) += A_R(0, t); // apply action
		X.col(t) = cart_track_step(X.col(t-1)); // advance env state

	}

	std::cout << X << std::endl;

	return 0;
}