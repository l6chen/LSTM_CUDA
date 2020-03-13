/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef INPUTGATE_H_
#define INPUTGATE_H_

#include <vector>
#include <iostream>
#include "basicLayer.h"

namespace inputGate {
	class InputGate: public basicLayer::BasicLayer {
	private:
        std::vector<std::vector<float>> Win;
        std::vector<float> bin;
		bool ifIn;
	public:
		InputGate(int embeds, int times, int hid, int cat, bool ifin) :
			basicLayer::BasicLayer(embeds,times,hid,cat), ifIn(ifin) {}
		void init() {}
	};
}

#endif /* INPUTGATE_H_ */
