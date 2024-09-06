#pragma once
#include <unistd.h>
#include <tuple>
#include <vector>

#include "torch/torch.h"
#include "env.hpp"

using namespace torch::indexing;

struct Trajectory
{
	struct Frame
	{
		torch::Tensor state;
		torch::Tensor action_probs;
		torch::Tensor action;
		unsigned action_idx;
		float reward;
	};

	static float R(const std::vector<Frame>& T, float gamma=0.999f)
	{
		float r = 0.0f;
		for (int i = 0; i < T.size(); i++)
		{
			r += pow(gamma, i) * T[i].reward;
		}

		return r;
	}
};

namespace policy
{
	struct Policy
	{
		// virtual bool load_params(const std::string& path) = 0;
		// virtual void save_params(const std::string& path) = 0;
		virtual void act(const std::vector<float>& x, Environment& env, std::vector<Trajectory::Frame>& traj) = 0;
		virtual void train(const std::vector<Trajectory::Frame>& traj, float learning_rate) = 0;
	};

	struct Discrete : public Policy, torch::nn::Module
	{
		Discrete();
		torch::Tensor forward(torch::Tensor x);

		virtual void act(const std::vector<float>& x, Environment& env, std::vector<Trajectory::Frame>& traj) override;
		virtual void train(const std::vector<Trajectory::Frame>& traj, float learning_rate) override;
	private:
		torch::nn::Linear l0 = nullptr, l1 = nullptr, l2 = nullptr, l3 = nullptr;
	};

	struct Continuous : public Policy, torch::nn::Module
	{
		Continuous();
		torch::Tensor forward(torch::Tensor x);

		virtual void act(const std::vector<float>& x, Environment& env, std::vector<Trajectory::Frame>& traj) override;
		virtual void train(const std::vector<Trajectory::Frame>& traj, float learning_rate) override;
	private:
		torch::nn::Linear l0 = nullptr, l1 = nullptr, l2 = nullptr;
	};

	struct hyper_parameters
	{
		unsigned epochs = 1;
		unsigned batch_size = 1;
		float learning_rate = 0.001f;
	};

	void init(size_t observation_size, size_t action_size);

	bool loaded();

	void save(const std::string& path);

	void train_policy_gradient(const std::vector<Trajectory::Frame>& traj, const hyper_parameters& hp={});

	torch::Tensor act_probs(torch::Tensor x);

	int act(torch::Tensor probs, bool stochastic=true);

}