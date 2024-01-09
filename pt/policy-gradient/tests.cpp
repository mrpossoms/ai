#include "env.hpp"
#include "net.hpp"
#include "rl.hpp"

#include <iostream>

bool near(float a, float b)
{
	return std::fabs(a - b) <= std::numeric_limits<float>::epsilon();
}

void check_positive_reinforcement()
{
	RL::Trajectory traj;

	auto probs = torch::rand({1, 4});
	probs[0][3] += 1;
	probs = torch::softmax(probs, 1);
	std::cout << probs << std::endl;

}

void check_policy_probabilities()
{
	net::init(4, 4);
	auto probs = net::act_probs(torch::rand({1, 4}));

	std::cout << probs << std::endl;

	assert(near(probs.sum().item<float>(), 1.f));
}

int main(int argc, char const *argv[])
{
	check_policy_probabilities();
	check_positive_reinforcement();

	return 0;
}