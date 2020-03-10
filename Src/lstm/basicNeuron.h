/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef BASICNEURON_H_
#define BASICNEURON_H_

#include <vector>
#include <iostream>

namespace basicNeuron {
	class BasicNeuron {
	private:
        std::vector<std::vector<float>> W;
        std::vector<float> b;

	public:
        BasicNeuron(std::vector<std::vector<float>> W, std::vector<float> b);
	};
}

#endif /* BASICNEURON_H_ */
