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
	protected:
		float* W, * b;
		float* WGrad, * bGrad;
		int Wlen, blen;
		int curTime;

	public:
		DenseLayer(int embeds, int times, int hid, int cat);
		~DenseLayer();

		void init() override;
		inline void WbGradInit();

		float* forward(float* x, float* h, void (*activate)(float* A, int n)) override;
		void calGrad(float* y, float* t, float* h);
		void updateWb(float lr);

		void showW() const;
		void showb() const;
		void showforward(float* in) const;

		float* getW() const { return W; }

		float loss(float* p, float* t);
	};
}
#endif /* DENSELAYER_H_ */