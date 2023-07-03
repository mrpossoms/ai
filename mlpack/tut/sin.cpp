#include <mlpack.hpp>
#include <ascii/ascii.h>

using namespace mlpack;

template<typename FNN>
void print_network_details(const FNN& net)
{
	unsigned i = 0;
	const auto& network = net.Network();
	for (const auto& layer : network)
	{
		std::cout << "layer" << i++ << ": " << layer->WeightSize() << " -> " << layer->OutputSize() << std::endl;
	}
}

int main (int argc, const char* argv[])
{
	const auto N = 1024;

	arma::mat X(1, N);
	arma::mat Y(1, N);

	for (unsigned i = 0; i < N; i++)
	{
		auto p = i / (float)N;
		auto t = p * 2 * M_PI;

		X.col(i)[0] = t;
		Y.col(i)[0] = sin(t);
	}

	// Step size of the optimizer.
	const double STEP_SIZE = 1e-2;
	// Number of data points in each iteration of SGD
	const size_t BATCH_SIZE = 32;//1024;
	// Allow up to 50 epochs, unless we are stopped early by EarlyStopAtMinLoss.
	const int EPOCHS = 100 * X.n_cols;


	FFN<HuberLoss> model;
	model.Add<Linear>(2);
	model.Add<Sigmoid>();
	model.Add<Linear>(2);
	model.Add<Linear>(1);

	// ens::Adam optimizer;
	ens::Adam optimizer(
		STEP_SIZE,  // Step size of the optimizer.
		BATCH_SIZE, // Batch size. Number of data points that are used in each
		            // iteration.
		0.9,        // Exponential decay rate for the first moment estimates.
		0.999, // Exponential decay rate for the weighted infinity norm estimates.
		1e-8,  // Value used to initialise the mean squared gradient parameter.
		EPOCHS, // Max number of iterations.
		1e-8,           // Tolerance.
		true);

	// Declare callback to store best training weights.
	ens::StoreBestCoordinates<arma::mat> bestCoordinates;

	// Train neural network. If this is the first iteration, weights are
	// random, using current values as starting point otherwise.
	model.Train(
		X,
		Y,
		optimizer,
		ens::PrintLoss(),
		ens::ProgressBar(),
		// Stop the training using Early Stop at min loss.
		ens::EarlyStopAtMinLoss(
		  [&](const arma::mat& /* param */)
		  {
		    double validationLoss = model.Evaluate(X, Y);
		    std::cout << "Validation loss: " << validationLoss << "."
		        << std::endl;
		    return validationLoss;
		  }),
		// Store best coordinates (neural network weights)
		bestCoordinates);

	std::cout << std::endl;

	print_network_details<>(model);

	std::unordered_map<std::string, std::vector<double>> traces;

	arma::mat h_mat(1, N), t_mat(1, N);

	for (unsigned i = 0; i < N; i++)
	{
		auto p = i / (float)N;
		auto t = p * 2 * M_PI;
		t_mat.row(0)[i] = t;
	}

	model.Predict(X, h_mat);

	for (unsigned i = 0; i < 100; i++)
	{
		traces["gt"].push_back(Y.col(i * 10)[0]);
		traces["h"].push_back(h_mat.col(i * 10)[0]);
		// traces["x"].push_back(X.col(i * 10)[0]);
	}	

	ascii::Asciichart chart(traces);
	std::cerr << chart.show_legend(true).offset(3).height(30).legend_padding(3).Plot();

	return 0;
}