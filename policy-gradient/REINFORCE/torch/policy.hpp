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
		torch::Tensor reward;
	};	

struct Trajectory
{
	Trajectory(int length, int observation_size, int action_size)
	{
		_states = torch::zeros({length, observation_size});
		_action_probs = torch::zeros({length, action_size});
		_actions = torch::zeros({length, action_size});
		_rewards = torch::zeros({length, 1});
		_capacity = length;
	}

	float R(float gamma=0.999f)
	{
		float r = 0.0f;
		for (int i = 0; i < this->size(); i++)
		{
			r += pow(gamma, i) * this->_rewards[i].item<float>();
		}

		return r;
	}

	Frame operator[](size_t idx) const
	{
		return Frame{
			_states[idx],
			_action_probs[idx],
			_actions[idx],
			(unsigned)_actions[idx].argmax().item<int>(),
			_rewards[idx]
		};
	}

	void push_back(const Frame& frame)
	{
		if (_size < _capacity)
		{
			// _states.index_put_({_size}, frame.state);
			// _action_probs.index_put_({_size}, frame.action_probs);
			// _actions.index_put_({_size}, frame.action);
			// _rewards.index_put_({_size}, frame.reward);
			_states[_size] = frame.state.flatten();
			_action_probs[_size] = frame.action_probs.flatten();
			_actions[_size] = frame.action.flatten();
			_rewards[_size] = frame.reward;
			_size++;
		}
	}

	void clear() { _size = 0; }

	const size_t size() const { return _size; }

private:
	torch::Tensor _states;
	torch::Tensor _action_probs;
	torch::Tensor _actions;
	torch::Tensor _rewards;
	size_t _size = 0;
	size_t _capacity = 0;
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
		virtual size_t action_size() = 0;
		virtual size_t observation_size() = 0;
	};

	struct Discrete : public Policy, torch::nn::Module
	{
		Discrete();
		torch::Tensor forward(torch::Tensor x);

		virtual void act(const std::vector<float>& x, Environment& env, trajectory::Trajectory& traj) override;
		virtual void train(const trajectory::Trajectory& traj, float learning_rate) override;
		virtual size_t action_size() override { return 4; }
		virtual size_t observation_size() override { return 4; }
	private:
		torch::nn::Linear l0 = nullptr, l1 = nullptr, l2 = nullptr, l3 = nullptr;
	};

	struct Continuous : public Policy, torch::nn::Module
	{
		Continuous();
		torch::Tensor forward(torch::Tensor x);

		virtual void act(const std::vector<float>& x, Environment& env, trajectory::Trajectory& traj) override;
		virtual void train(const trajectory::Trajectory& traj, float learning_rate) override;
		virtual size_t action_size() override { return 4; }
		virtual size_t observation_size() override { return 4; }

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