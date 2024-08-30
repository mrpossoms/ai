#include <armadillo>
#include <cassert>
#include "policy_gradient.hpp"

arma::mat cart_track_step(const arma::subview_col<double>& x_t, double u_t)
{
	arma::mat stm({
		{1, 0.1},
		{0, 0.99},
	});
	arma::mat u(2, 1, arma::fill::zeros);
	u.row(1)[0] = u_t;
	return stm * (x_t + u);
}

double R_t(const arma::subview_col<double>& x_t, const arma::subview_col<double>& x_t1)
{
	return pow(x_t[0], 2.0) - pow(x_t1[0], 2.0);
}

arma::mat numerical_grad(const arma::subview_col<double>& x_t, const arma::subview_col<double>& A_R, arma::mat W)
{
	double e = 1e-8;

	arma::mat g;
	g.set_size(size(W));

	for (unsigned r = 0; r < W.n_rows; r++)
	for (unsigned c = 0; c < W.n_cols; c++)
	{
		arma::mat eps(g.n_rows, g.n_cols, arma::fill::zeros);
		eps(r, c) = e;

		arma::mat u_0 = (x_t * W); // compute action
		arma::mat x_t1_0 = cart_track_step(x_t, u_0[0]);
		arma::mat u_1 = (x_t * (W + eps)); // compute action
		arma::mat x_t1_1 = cart_track_step(x_t, u_1[0]);

		g(r, c) = (R_t(x_t, x_t1_1.col(0)) - R_t(x_t, x_t1_0.col(0))) / e;
	}

	return g * A_R[1];

}

int main(int argc, char const *argv[])
{
	PolicyGradientLoss pg;

	arma::mat X0(2, 1, arma::fill::randn);
	arma::mat X(2, 10, arma::fill::zeros);
	arma::mat A_R(2, 10);
	arma::mat W(1, 2, arma::fill::randn);

	X.col(0) = X0;


	// for (unsigned eps = )
	for (unsigned t = 1; t < X.n_cols; t++)
	{
		arma::mat u = (X.col(t-1) * W); // compute action
		X(1, t-1) += A_R(0, t-1); // apply action
		X.col(t) = cart_track_step(X.col(t-1), u(0, 0)); // advance env state
		A_R(0, t-1) = u(0, 0);
		A_R(1, t-1) = R_t(X.col(t-1), X.col(t)); // compute reward
	}

	std::cout << "States: " << std::endl << X << std::endl;
	std::cout << "Actions: " << std::endl << A_R.row(0) << std::endl;
	std::cout << "Rewards: " << std::endl << A_R.row(1) << std::endl;

	std::cout << "num G: " <<  numerical_grad(X0.col(0), A_R.col(0), W) << std::endl;

	arma::mat G;
	pg.Backward(X0.col(0), A_R.col(0), G);
	std::cout << "analy G: " <<  G << std::endl;

	return 0;
}