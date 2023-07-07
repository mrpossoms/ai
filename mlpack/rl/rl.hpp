#pragma once
// #include <mlpack/core.hpp>
#include <armadillo>

namespace RL 
{
	struct State
	{
		float d_goal[2];
		float vel[2];
		float vel3[2];
	};

	struct Action
	{
		float d_r;
		float d_c;
	};


	struct Trajectory
	{
		arma::fmat states;
		arma::fmat action_rewards;
		unsigned write_ptr = 0;

		Trajectory(size_t len=128);

		void append(const State& x, const Action& a, float r);

		float R(float gamma=0.999f);

		bool full() const { return write_ptr >= states.n_cols; }

		void clear() { write_ptr = 0; }
	};

	struct ReplayBuffer
	{
		std::vector<Trajectory> trajectories;
		unsigned write_ptr = 0;

		ReplayBuffer(size_t len=16);

		void append(const State& x, const Action& a, float r);

		std::vector<Trajectory*> sample(size_t batch_size=128);

		bool full() const { return write_ptr >= trajectories.size(); }

		void clear()
		{
			for (auto& t : trajectories)
				t.clear();
			write_ptr = 0; 
		}

		Trajectory& current_trajectory() { return trajectories[write_ptr % trajectories.size()]; }

		Trajectory& last_trajectory() { return trajectories[(write_ptr-1) % trajectories.size()]; }

		float avg_reward()
		{
			float r = 0.0f;
			for (auto& t : trajectories)
			{
				r += t.R();
			}

			return r / (float)trajectories.size();
		}
	};

}