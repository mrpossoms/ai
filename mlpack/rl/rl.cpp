#include "rl.hpp"

RL::Trajectory::Trajectory(size_t len)
{
	states = arma::fmat(4, len);
	action_rewards = arma::fmat(3, len);
}

void RL::Trajectory::append(const RL::State& x, const RL::Action& a, float r)
{
	write_ptr %= states.n_cols;

	states.col(write_ptr)[0] = x.d_goal[0];
	states.col(write_ptr)[1] = x.d_goal[1];
	states.col(write_ptr)[2] = x.vel[0];
	states.col(write_ptr)[3] = x.vel[1];

	action_rewards.col(write_ptr)[0] = a.d_r;
	action_rewards.col(write_ptr)[1] = a.d_c;
	action_rewards.col(write_ptr)[2] = r;

	write_ptr++;
}

float RL::Trajectory::R(float gamma)
{
	float r = 0.0f;
	for (int i = 0; i < write_ptr; i++)
	{
		r += pow(gamma, i) * action_rewards.col(i)[action_rewards.n_rows - 1];
	}

	return r;
}

RL::ReplayBuffer::ReplayBuffer(size_t len)
{
	trajectories = std::vector<Trajectory>(len);
}

void RL::ReplayBuffer::append(const RL::State& x, const RL::Action& a, float r)
{
	if (trajectories[write_ptr % trajectories.size()].full())
	{
		write_ptr++;
		trajectories[write_ptr % trajectories.size()].clear();
	}

	trajectories[write_ptr % trajectories.size()].append(x, a, r);
}

std::vector<RL::Trajectory*> RL::ReplayBuffer::sample(size_t batch_size)
{
	std::vector<Trajectory*> batch;
	for (int i = 0; i < batch_size; i++)
	{
		batch.push_back(&trajectories[rand() % trajectories.size()]);
	}

	return batch;
}