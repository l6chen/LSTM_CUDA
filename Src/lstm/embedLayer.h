/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef EMBEDLAYER_H_
#define EMBEDLAYER_H_

#include <vector>
#include <string>
#include <iostream>
#include "basicLayer.h"
#include "util.h"

namespace embedLayer {
	class EmbedLayer : public basicLayer::BasicLayer {
	protected:
		float* W;
		float* WGrad;
		int Wlen, dictSize;

	public:
		EmbedLayer(int embeds, int times, int hid, int cat, int dic);
		~EmbedLayer();

		void init() override;
		inline void WbGradInit();

		float* forward(int textCode);
		void calGrad(float* delta, int textCode);
		void updateWb(float lr);

		void showW() const;

		void showforward(float* in) const;

		float* getW() const { return W; }

	};
}
#endif /* EMBEDLAYER_H_ */