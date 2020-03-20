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
	public:
        float* Wh, * Wx, * b;
		float* WhGrad, * WxGrad, * bGrad;
		int Whlen, Wxlen, blen;
		int curTime = 0;


	
		GateLayer(int embeds, int times, int hid, int cat, float lr);
		~GateLayer();

		void init() override;
		inline void WbGradInit();

		float* forward(float* x, float* h, float* (*activate)(float* A, int n))override;
		void calGrad(float* x, basicLayer::OutputsDelta* datas, std::vector<float*>* dgates)override;
		void updateWb()override;

		void showW() const;
		void showb() const;
		void showforward(float* in) const;

		float* getWh() const override{ return Wh; }
		float* getWx() const override{ return Wx; }

		void checkGrad();

	};

	/*******************************Four Gates with Parameter***********************************/
	class OutputGate : public GateLayer {
	public:
		OutputGate(int embeds, int times, int hid, int cat, float lr) :
			GateLayer(embeds, times, hid, cat, lr) { }

		void calDeltak(basicLayer::OutputsDelta* datas, int k) override;
		~OutputGate() {};
	};

	class InputGate : public GateLayer {
	public:
		InputGate(int embeds, int times, int hid, int cat, float lr) :
			GateLayer(embeds, times, hid, cat, lr) {}

		void calDeltak(basicLayer::OutputsDelta* datas, int k) override;
		~InputGate() {};
	};

	class ForgetGate : public GateLayer {
	public:
		ForgetGate(int embeds, int times, int hid, int cat, float lr) :
			GateLayer(embeds, times, hid, cat, lr) {}

		void calDeltak(basicLayer::OutputsDelta* datas, int k) override;
		~ForgetGate() {};
	};

	class CellTGate : public GateLayer {
	public:
		CellTGate(int embeds, int times, int hid, int cat, float lr) :
			GateLayer(embeds, times, hid, cat, lr) {}

		void calDeltak(basicLayer::OutputsDelta* datas, int k) override;
		~CellTGate() {};
	};

	/***********************************Special Layers***************************************/

	class CellGate : public basicLayer::BasicLayer {
	public:
		int curTime = 0;
		CellGate(int embeds, int times, int hid, int cat, float lr) :
			basicLayer::BasicLayer(embeds, times, hid, cat, lr) {}
		float* forward(basicLayer::OutputsDelta* datas) override;
		void resetTime() override { curTime = 0; }
	};

	class HiddenGate : public basicLayer::BasicLayer {
	public:
		int curTime = 0;
		HiddenGate(int embeds, int times, int hid, int cat, float lr) :
			basicLayer::BasicLayer(embeds, times, hid, cat, lr) {}
		float* forward(basicLayer::OutputsDelta* datas) override;
		void resetTime() override { curTime = 0; }
	};
}

#endif /* GATELAYER_H_ */
