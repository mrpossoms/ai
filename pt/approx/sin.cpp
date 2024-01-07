#include <torch/torch.h>
#include <ascii/ascii.h>

struct FFN : torch::nn::Module
{
	FFN()
	{
		fc[0] = register_module("fc1", torch::nn::Linear(1, 3));
		fc[1] = register_module("fc2", torch::nn::Linear(3, 3));
		fc[2] = register_module("fc3", torch::nn::Linear(3, 1));
		fc[3] = register_module("fc4", torch::nn::Linear(1, 1));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::sigmoid(fc[0]->forward(x));
		x = fc[1]->forward(x);
		x = fc[2]->forward(x);
		x = fc[3]->forward(x);
		return x;
	}

	torch::nn::Linear fc[4] = {{nullptr}, {nullptr}, {nullptr}, {nullptr}};
};

void print_network_details(const FFN& model)
{
	for (auto& p : model.parameters())
	{
		std::cout << "-------------" << std::endl;
		std::cout << p << std::endl;
	}
}

int main (int argc, const char* argv[])
{
	const auto N = 1024;

	auto X = torch::zeros({N, 1});
	auto Y = torch::zeros({N, 1});

	for (unsigned i = 0; i < N; i++)
	{
		auto p = i / (float)N;
		auto t = p * 2 * M_PI;

		X[i] = t;
		Y[i] = sin(t);
	}

	// Step size of the optimizer.
	const double STEP_SIZE = 1e-2;
	// Number of data points in each iteration of SGD
	const size_t BATCH_SIZE = 1;//1024;
	// Allow up to 50 epochs, unless we are stopped early by EarlyStopAtMinLoss.
	const int EPOCHS = 10 * X.size(0);

	FFN model;

	torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(STEP_SIZE));

	for (int i = 0; i < EPOCHS; i++)
	{
		// Reset gradients.
		optimizer.zero_grad();
		// Forward pass.
		auto output = model.forward(X);
		// Calculate loss.
		auto loss = torch::mse_loss(output, Y);
		// Backward pass.
		loss.backward();
		// Update parameters.
		optimizer.step();

		if (i % 100 == 0)
		std::cout << "Epoch: " << i << " Loss: " << loss.item<float>() << std::endl;
	}

	print_network_details(model);

	std::unordered_map<std::string, std::vector<double>> traces;

	auto t_mat = torch::zeros(N);

	for (unsigned i = 0; i < N; i++)
	{
		auto p = i / (float)N;
		auto t = p * 2 * M_PI;
		t_mat[i] = t;
	}

	auto h_mat = model.forward(X);

	for (unsigned i = 0; i < 100; i++)
	{
		traces["gt"].push_back(Y[i * 10].item<float>());
		traces["h"].push_back(h_mat[i * 10].item<float>());
	}	

	ascii::Asciichart chart(traces);
	std::cerr << chart.show_legend(true).offset(3).height(30).legend_padding(3).Plot();

	return 0;
}
