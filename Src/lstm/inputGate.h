/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef INPUTGATE_H_
#define INPUTGATE_H_

#include <vector>
#include <iostream>

namespace inputGate {
	class InputGate {
	private:
        std::vector<std::vector<float>> W;
        std::vector<float> b;

	public:
        InputGate(std::vector<std::vector<float>> W, std::vector<float> b);
	};
}

#endif /* INPUTGATE_H_ */
