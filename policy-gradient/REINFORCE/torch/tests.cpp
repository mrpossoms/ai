#include "env.hpp"
#include "policy.hpp"

#include <iostream>
#include <fstream>

void write_csv(const std::string& filename, const std::map<std::string, std::vector<float>>& data)
{
	std::ofstream file(filename, std::ofstream::out);
	size_t rows = 0;

	for (auto& kvp : data) {
		if (rows == 0) {
			rows = kvp.second.size();
		}
		else {
			assert(rows == kvp.second.size());
		}

		file << kvp.first << ",";
	}
	file << std::endl;

	for (size_t i = 0; i < rows; i++) {
		for (auto& kvp : data) {
			file << kvp.second[i] << ",";
		}
		file << std::endl;
	}

	file.close();
}


bool near(float a, float b)
{
	return std::fabs(a - b) <= std::numeric_limits<float>::epsilon();
}

void check_policy_probabilities_discrete()
{
	policy::Discrete policy;
	Environment env;
	trajectory::Trajectory traj(1, policy.observation_size(), policy.action_size());
	assert(near(policy.act(env, traj).sum().item<float>(), 1.f));
}

void check_policy_probabilities_continuous()
{
	for (unsigned e = 0; e < 100; e++)
	{
		policy::Continuous policy;
		Environment env;
		trajectory::Trajectory traj(1, policy.observation_size(), policy.action_size());
		
		auto a = policy.act(env, traj); //.index({Slice(0, traj.size()), Slice(0, 2)});
		for(unsigned i = 0; i < policy.action_size(); i++)
		{
			float a_i = a.flatten()[i].item<float>();
			std::cout << "a" << std::to_string(i) << ": " << a_i << std::endl;
			assert(a.flatten()[i].item<float>() < 1.f && a.flatten()[i].item<float>() > 0.f);
		}		
	}
}


struct Dummy : public policy::Policy
{
	Dummy()
	{
		layer = register_module("l0", torch::nn::Linear(1, 2));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		return layer->forward(x);
	}

	virtual const torch::Tensor act(Environment& env, trajectory::Trajectory& traj) override
	{
		return forward(torch::ones({1, 1}));
	}
	
	torch::Tensor action_probabilities(const torch::Tensor& a_dist_params, const torch::Tensor& a)
	{
		// constexpr auto sqrt_2pi = std::sqrt(2 * M_PI);
		auto mu = a_dist_params.index({0, Slice(0, action_size())});
		auto sigma = action_sigma(a_dist_params);
		auto var = sigma.pow(2);

		// auto eps = 1e-3;
		// auto a_eps = torch::ones_like(a) * eps;
		// return torch::abs(gaussian(a + a_eps, mu, var) - gaussian(a, mu, var)) * eps;

		// return torch::clamp(policy::gaussian(a, mu, var), 0.0001f, 0.9999f);

		return policy::gaussian(a, mu, var);
	}

	torch::Tensor action_sigma(const torch::Tensor& a_dist_params)
	{
		// return torch::ones({1}) * 0.445f;
		return torch::log(torch::exp(a_dist_params.index({0, Slice(action_size(), output_size())})) + 1) + 0.01f;
	}

	virtual void train(const trajectory::Trajectory& traj, float learning_rate) override{}
	virtual long output_size() { return 2; }
	virtual long action_size() override { return 1; }
	virtual long observation_size() override { return 1; }

	torch::nn::Linear layer = { nullptr };
};

void plot_func_and_gradients(Dummy& policy, int iteration)
{
	constexpr auto min_x = -3.f;
	constexpr auto max_x = 3.f;

	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> dpr;
	std::vector<float> w0, w1;
	std::vector<float> gw0, gw1;
	std::vector<float> b0, b1;
	std::vector<float> gb0, gb1;

	for (int i = 0; i < 100; i++)
	{
		trajectory::Trajectory traj(1, policy.observation_size(), policy.action_size());

		auto x_0 = torch::ones({1, 1});
		auto x_t = torch::ones({1, 1}) * (i * ((max_x - min_x) / 100.f) + min_x);
		auto r = torch::ones({1});
		auto y_t = policy.forward(x_t);
		auto pr = policy.action_probabilities(y_t, x_t);
		pr.retain_grad();

		traj.push_back({x_0, pr, x_t, 0, r});
		policy.zero_grad();
		policy::Continuous::train(traj, policy, 0.0f);

		x.push_back(x_t.item<float>());
		y.push_back(pr.item<float>());
		
		dpr.push_back(pr.grad()[0].item<float>());

		for (const auto& param_pair : policy.named_parameters()) {
			auto& name = param_pair.key();
			auto& param = param_pair.value();
			if (name == "l0.weight") {
				w0.push_back(param[0].item<float>());
				gw0.push_back(param.grad()[0].item<float>());
				w1.push_back(param[1].item<float>());
				gw1.push_back(param.grad()[1].item<float>());
			}
			else if (name == "l0.bias") {
				gb0.push_back(param.grad()[0].item<float>());
				b0.push_back(param[0].item<float>());
				gb1.push_back(param.grad()[1].item<float>());
				b1.push_back(param[1].item<float>());
			}
		}
	}

	write_csv("/tmp/plot" + std::to_string(iteration) + ".csv", {
		{"x", x}, 
		{"pr", y}, 
		{"dpr", dpr}, 
		{"gw0", gw0}, 
		{"w0", w0}, 
		{"gw1", gw1}, 
		{"w1", w1}, 
		{"gb0", gb0}, 
		{"b0", b0}, 
		{"gb1", gb1},
		{"b1", b1},
	});
}



void check_policy_optimization_continuous()
{
	// policy::Continuous policy;
	Dummy policy;
	Environment env;
	trajectory::Trajectory traj(1, policy.observation_size(), policy.action_size());

	std::vector<float> probabilities;
	std::vector<float> sigmas;

	for (int i = 0; i < 100; i++)
	{
		std::cout << "======================" << std::endl;
		std::cout << "i: " << i << std::endl;
		auto x = torch::ones({1, policy.observation_size()});
		auto y = policy.forward(x);
		std::cout << "y: " << y << std::endl;

		auto a = torch::ones({1, policy.action_size()});
		auto r = torch::ones({1});
		auto pr = policy.action_probabilities(y, a);
		auto sigma = policy.action_sigma(y);
		auto mu = y.index({0, Slice(0, policy.action_size())});
		probabilities.push_back(pr[0][0].item<float>());
		sigmas.push_back(sigma[0].item<float>());
		std::cout << "pr: " << pr[0][0].item<float>() << " mu: " << mu[0].item<float>() << " sig: " << sigmas[sigmas.size()-1] << std::endl;
		traj.push_back({x, pr, a, 0, r});
		policy::Continuous::train(traj, policy, 0.1f);
		plot_func_and_gradients(policy, i);

		traj.clear();
		std::cout << "======================" << std::endl;
	}

	for (int i = 0; i < probabilities.size(); i++)
	{
		std::cout << "probabilities[" << std::to_string(i) << "]: " << probabilities[i] << std::endl;

		assert(probabilities[i] >= probabilities[i - 1]);
	}
	assert(probabilities[0] < probabilities[probabilities.size()-1]);

	// auto a = policy.act(env, traj).index({Slice(0, traj.size()), Slice(0, 2)});
	// auto old_probs = a.clone();
	// auto reward = torch::ones({1, 1});
	// auto reward_t = torch::from_blob(&reward, {1}, torch::kFloat).clone();
	// traj.push_back({torch::ones({1, 1}), a, a, 0, reward_t});
	// policy.train(traj, 0.1f);
	// auto new_probs = policy.act(env, traj).index({Slice(0, traj.size()), Slice(0, 2)});
	// assert((new_probs - old_probs).sum().item<float>() > 0);
}

void check_positive_reinforcement()
{
	Dummy dummy_net;

	auto input = torch::ones({1, 1});
	auto probs = dummy_net.forward(input);
	
	probs = torch::softmax(probs, 1);
	std::cout << "------------" << std::endl;

	auto old_probs = probs.clone();

	probs[0][0].backward();

	for (auto& param : dummy_net.parameters())
	{
		param.data() += param.grad() * 0.1f;
	}

	probs = dummy_net.forward(torch::ones({1, 1}));
	probs = torch::softmax(probs, 1);

	std::cout << "old probs: " << old_probs << std::endl;
	std::cout << "new probs: " << probs << std::endl;
		
	assert(probs[0][0].item<float>() >= old_probs[0][0].item<float>());
}

void check_sanity()
{
	float a[] = { 1, 0.5f, 2, 0.25f };
	auto x = torch::from_blob(a, {2, 2}, torch::kFloat).clone();
	std::cout << "x: " << x << std::endl;
	auto p = x.prod(1);
	std::cout << "p: " << p << std::endl;
	assert(p[0].item<float>() == 0.5f);
	assert(p[1].item<float>() == 0.5f);
}

int main(int argc, char const *argv[])
{
	check_sanity();
	check_policy_probabilities_discrete();
	// check_policy_probabilities_continuous();
	check_positive_reinforcement();
	check_policy_optimization_continuous();
	
	return 0;
}
