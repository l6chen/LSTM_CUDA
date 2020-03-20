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
	public:
		float* W;
		float* WGrad;
		int Wlen, dictSize;

	
		EmbedLayer(int embeds, int times, int hid, int cat, float lr, int dic);
		~EmbedLayer();

		void init() override;
		inline void WbGradInit();
		
		float* forward(int textCode)override;
		
		void calGrad(float* delta, int textCode)override;
		void updateWb()override;

		void showW() const;

		void showforward(float* in) const;

		float* getW() const { return W; }

	};
}
#endif /* EMBEDLAYER_H_ */