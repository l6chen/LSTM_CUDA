/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef OUTPUTGATE_H_
#define OUTPUTGATE_H_

#include <vector>
#include <iostream>

namespace outputGate {
	class OutputGate {
	private:
        std::vector<std::vector<float>> W;
        std::vector<float> b;

	public:
        OutputGate(std::vector<std::vector<float>> W, std::vector<float> b);
	};
}

#endif /* OUTPUTGATE_H_ */
