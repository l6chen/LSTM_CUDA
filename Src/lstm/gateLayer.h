/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef GATELAYER_H_
#define GATELAYER_H_

#include <vector>
#include <string>
#include <iostream>
#include "basicLayer.h"
#include "util.h"

namespace gateLayer {
	class GateLayer: public basicLayer::BasicLayer {
	private:
        float* Wh, * Wx, * b;
		int Whlen, Wxlen, blen;
		void updateWb(float lr, float* Wh_grad, float* Wx_grad, float* b_grad);
		float* calLoss(float* dh, float* c, float* out) const;
		float* calWhGrad(float* dhpre, float* delta) const;
		float* calWxGrad(float* x, float* out) const;
		float* calbGrad(float* deltaf) const;
		std::string gatename;

	public:
		GateLayer(int embeds, int times, int hid, int cat, std::string gt) :
		basicLayer::BasicLayer(embeds, times, hid, cat), gatename(gt) { init(); }
		void init() override;
		float* forward(float* h, float* x) override;
		float* backward(float* dh, float* x) override;
		void showW() const;
		void showb() const;
		void showforward(float* in) const;
		~GateLayer();
	};
}

#endif /* GATELAYER_H_ */
