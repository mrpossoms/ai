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
		torch::Tensor output;
		torch::Tensor action_probs;
		torch::Tensor action;
		unsigned action_idx;
		torch::Tensor reward;
	};	

struct Trajectory
{
	Trajectory(int length, int observation_size, int action_size, int output_size)
	{
		states = torch::zeros({length, observation_size});
		outputs = torch::zeros({length, output_size});
		action_probs = torch::zeros({length, action_size});
		actions = torch::zeros({length, action_size});
		rewards = torch::zeros({length, 1});
		_capacity = length;
	}

	float R(float gamma=0.999f)
	{
		float r = 0.0f;
		for (int i = 0; i < this->size(); i++)
		{
			r += pow(gamma, i) * this->rewards[i].item<float>();
		}

		return r;
	}

	Frame operator[](size_t idx) const
	{
		return Frame{
			states[idx],
			outputs[idx],
			action_probs[idx],
			actions[idx],
			(unsigned)actions[idx].argmax().item<int>(),
			rewards[idx]
		};
	}

	void push_back(const Frame& frame)
	{
		if (_size < _capacity)
		{
			// states.index_put_({_size}, frame.state);
			// action_probs.index_put_({_size}, frame.action_probs);
			// actions.index_put_({_size}, frame.action);
			// rewards.index_put_({_size}, frame.reward);
			states[_size] = frame.state.flatten();
			outputs[_size] = frame.output.flatten();
			action_probs[_size] = frame.action_probs.flatten();
			actions[_size] = frame.action.flatten();
			rewards[_size] = frame.reward;
			_size++;
		}
	}

	void clear()
	{ 
		_size = 0;
		states = torch::zeros({_capacity, states.size(1)});
		action_probs = torch::zeros({_capacity, action_probs.size(1)});
		actions = torch::zeros({_capacity, action_probs.size(1)});
		rewards = torch::zeros({_capacity, 1});
	}

	const size_t size() const { return _size; }

	torch::Tensor states;
	torch::Tensor outputs;
	torch::Tensor action_probs;
	torch::Tensor actions;
	torch::Tensor rewards;
private:
	size_t _size = 0;
	long _capacity = 0;
};

}


namespace policy
{
	struct Policy : public torch::nn::Module
	{
		// virtual bool load_params(const std::string& path) = 0;
		// virtual void save_params(const std::string& path) = 0;
		virtual torch::Tensor forward(torch::Tensor x) = 0;
		virtual const torch::Tensor act(Environment& env, trajectory::Trajectory& traj) = 0;
		virtual void train(const trajectory::Trajectory& traj, float learning_rate) = 0;
		virtual long output_size() { return action_size(); }
		virtual long action_size() = 0;
		virtual long observation_size() = 0;
		void print_params()
		{
			for (auto& param_pair : named_parameters())
			{
				auto& name = param_pair.key();
				auto& param = param_pair.value();
				std::cout << name << std::endl;
				std::cout << param << std::endl;
			}
		}
	};

	struct Discrete : public Policy
	{
		Discrete();
		virtual torch::Tensor forward(torch::Tensor x) override;

		virtual const torch::Tensor act(Environment& env, trajectory::Trajectory& traj) override;
		virtual void train(const trajectory::Trajectory& traj, float learning_rate) override;
		virtual long action_size() override { return 4; }
		virtual long observation_size() override { return 4; }
	private:
		torch::nn::Linear l0 = nullptr, l1 = nullptr, l2 = nullptr, l3 = nullptr;
	};

	struct Continuous : public Policy
	{
		Continuous();
		torch::Tensor action_sigma(const torch::Tensor& a_dist_params);
		torch::Tensor action_probabilities(const torch::Tensor& a_dist_params, const torch::Tensor& a);

		static torch::Tensor tensor_from_state(Environment& env);
		static void train(const trajectory::Trajectory& traj, Policy& policy, float learning_rate);
		static torch::Tensor action_probabilities(const torch::Tensor& a, const torch::Tensor& mu, const torch::Tensor& var);

		virtual torch::Tensor forward(torch::Tensor x) override;		
		virtual const torch::Tensor act(Environment& env, trajectory::Trajectory& traj) override;
		virtual void train(const trajectory::Trajectory& traj, float learning_rate) override;
		virtual long action_size() override { return 2; }
		virtual long output_size() override { return 4; }
		virtual long observation_size() override { return 4; }

	private:
		torch::nn::Linear l0 = nullptr, l1 = nullptr, l2 = nullptr;
	};

	torch::Tensor gaussian(const torch::Tensor& x, const torch::Tensor& mu, const torch::Tensor& var);
}
