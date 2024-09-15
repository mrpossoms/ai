#pragma once
#include <unistd.h>
#include <tuple>
#include <vector>

#include "torch/torch.h"
#include "env.hpp"

using namespace torch::indexing;

namespace trajectory
{
	struct Frame
	{
		torch::Tensor state;
		torch::Tensor action_probs;
		torch::Tensor action;
		unsigned action_idx;
		float reward;
	};	

struct Trajectory : public std::vector<Frame>
{
	Trajectory() = default;

	Trajectory(const std::vector<Frame>& T) : std::vector<Frame>(T) {}

	Trajectory(const Trajectory& T) : std::vector<Frame>(T) {}

	Trajectory(const Trajectory&& T) : std::vector<Frame>(T) {}

	Trajectory& operator=(const Trajectory& T)
	{
		std::vector<Frame>::operator=(T);
		return *this;
	}

	Trajectory& operator=(const Trajectory&& T)
	{
		std::vector<Frame>::operator=(T);
		return *this;
	}

	float R(float gamma=0.999f)
	{
		float r = 0.0f;
		for (int i = 0; i < this->size(); i++)
		{
			r += pow(gamma, i) * (*this)[i].reward;
		}

		return r;
	}
};

}


namespace policy
{
	struct Policy
	{
		// virtual bool load_params(const std::string& path) = 0;
		// virtual void save_params(const std::string& path) = 0;
		virtual void act(const std::vector<float>& x, Environment& env, trajectory::Trajectory& traj) = 0;
		virtual void train(const trajectory::Trajectory& traj, float learning_rate) = 0;
	};

	struct Discrete : public Policy, torch::nn::Module
	{
		Discrete();
		torch::Tensor forward(torch::Tensor x);

		virtual void act(const std::vector<float>& x, Environment& env, trajectory::Trajectory& traj) override;
		virtual void train(const trajectory::Trajectory& traj, float learning_rate) override;
	private:
		torch::nn::Linear l0 = nullptr, l1 = nullptr, l2 = nullptr, l3 = nullptr;
	};

	struct Continuous : public Policy, torch::nn::Module
	{
		Continuous();
		torch::Tensor forward(torch::Tensor x);

		virtual void act(const std::vector<float>& x, Environment& env, trajectory::Trajectory& traj) override;
		virtual void train(const trajectory::Trajectory& traj, float learning_rate) override;
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

	void train_policy_gradient(const trajectory::Trajectory& traj, const hyper_parameters& hp={});

	torch::Tensor act_probs(torch::Tensor x);

	int act(torch::Tensor probs, bool stochastic=true);

}