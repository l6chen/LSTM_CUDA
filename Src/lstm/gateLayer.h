/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef GATELAYER_H_
#define GATELAYER_H_

#include <vector>
#include <iostream>
#include "basicLayer.h"
#include "util.h"

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

namespace gateLayer {
	class GateLayer: public basicLayer::BasicLayer {
	private:
        float* Win;
        float* bin;
		int curStep;
		void updateWb();
		void calLoss(float* dh, float* x) const;
		void calGrad(float* dh, float* x) const;

	public:
		GateLayer(int embeds, int times, int hid, int cat) :
			basicLayer::BasicLayer(embeds, times, hid, cat) { init(); }
		void init() override;
		float* forward(float* h, float* x) override;
		float* backward(float* dh, float* x) override;
		void showW() const;
		void showb() const;
		void showforward(float* in) const;
	};
}

#endif /* GATELAYER_H_ */
