/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef MODULEGATE_H_
#define MODULEGATE_H_

#include <vector>
#include <iostream>

namespace moduleGate {
	class ModuleGate {
	private:
        std::vector<std::vector<float>> W;
        std::vector<float> b;

	public:
        ModuleGate(std::vector<std::vector<float>> W, std::vector<float> b);
	};
}

#endif /* MODULEGATE_H_ */
