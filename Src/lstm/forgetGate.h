/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef FORGETGATE_H_
#define FORGETGATE_H_

#include <vector>
#include <iostream>

namespace forgetGate {
	class ForgetGate {
	private:
        std::vector<std::vector<float>> W;
        std::vector<float> b;

	public:
        ForgetGate(std::vector<std::vector<float>> W, std::vector<float> b);
	};
}

#endif /* FORGETGATE_H_ */
