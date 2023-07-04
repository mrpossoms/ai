#pragma once
#include "structs.hpp"

namespace net 
{
	void train(const std::vector<std::tuple<state, action, double>>& trajectory);

	action act(const state& x);
}
