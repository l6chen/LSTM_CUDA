/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <vector>
#include <iostream>
#include "basicLayer.h"

namespace basicLayer {
	float* BasicLayer::concatVec(float* vecA, float* vecB,
		const int a, const int b) {
		int newsize = a + b;
		float* out = new float[newsize];
		if (out != NULL) {
			for (int i = 0; i < newsize; i++) {
				if (i < a)
					out[i] = vecA[i];
				else
					out[i] = vecB[i - a];
			}
		}
		return out;
	}
	void BasicLayer::showVar() const {
		for (int i = 0; i < 4; i++) {
			std::cout << allVar[i];
		}
	}
}