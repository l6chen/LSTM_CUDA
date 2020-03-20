/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef DENSELAYER_H_
#define DENSELAYER_H_

#include <vector>
#include <string>
#include <iostream>
#include "basicLayer.h"
#include "util.h"

namespace denseLayer {
	class DenseLayer : public basicLayer::BasicLayer {
	public:
		float* W, * b;
		float* WGrad, * bGrad;
		int Wlen, blen;
		void calGrad(float* delta, float* h);
		void updateWb(float lr);
		

	
		DenseLayer(int embeds, int times, int hid, int cat, float lr);
		~DenseLayer();

		void init() override;
		inline void WbGradInit();

		float* forward(float* x, float* h, float* (*activate)(float* A, int n))override;
		float* backward(float* h, float* y, float* t)override;//return delta for last Layer

		void showW() const;
		void showb() const;
		void showforward(float* in) const;
		float calLoss(float* p, float* t)override;
		float* getW() const { return W; }

		void showGrad();
		

	};
}
#endif /* DENSELAYER_H_ */