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
	protected:
        float* Wh, * Wx, * b;
		float* WhGrad, * WxGrad, * bGrad;
		int Whlen, Wxlen, blen;
		int curTime;

	public:
		GateLayer(int embeds, int times, int hid, int cat);
		~GateLayer();

		void init() override;
		inline void WbGradInit();

		float* forward(float* x, float* h, void (*activate)(float* A, int n)) override;
		void calGrad(float* x, basicLayer::OutputsDelta& datas, std::vector<float*> dgates);
		void updateWb(float lr);

		void showW() const;
		void showb() const;
		void showforward(float* in) const;

		float* getWh() const { return Wh; }
	};

	/*******************************Four Gates with Parameter***********************************/
	class OutputGate : public GateLayer {
	public:
		OutputGate(int embeds, int times, int hid, int cat) :
			GateLayer(embeds, times, hid, cat) { }

		void calDelta(float* x, basicLayer::OutputsDelta& datas);
		~OutputGate() {};
	};

	class InputGate : public GateLayer {
	public:
		InputGate(int embeds, int times, int hid, int cat) :
			GateLayer(embeds, times, hid, cat) {}

		void calDelta(float* x, basicLayer::OutputsDelta& datas);
		~InputGate() {};
	};

	class ForgetGate : public GateLayer {
	public:
		ForgetGate(int embeds, int times, int hid, int cat) :
			GateLayer(embeds, times, hid, cat) {}

		void calDelta(float* x, basicLayer::OutputsDelta& datas);
		~ForgetGate() {};
	};

	class CellTGate : public GateLayer {
	public:
		CellTGate(int embeds, int times, int hid, int cat) :
			GateLayer(embeds, times, hid, cat) {}

		void calDelta(float* x, basicLayer::OutputsDelta& datas);
		~CellTGate() {};
	};

	/***********************************Special Layers***************************************/

	class CellGate : public basicLayer::BasicLayer {
	private:
		int curTime;
	public:
		CellGate(int embeds, int times, int hid, int cat) :
			basicLayer::BasicLayer(embeds, times, hid, cat) {}
		float* forward(basicLayer::OutputsDelta datas);
	};

	class HiddenGate : public basicLayer::BasicLayer {
	private:
		int curTime;
	public:
		HiddenGate(int embeds, int times, int hid, int cat) :
			basicLayer::BasicLayer(embeds, times, hid, cat) {}
		float* forward(basicLayer::OutputsDelta datas);
	};
}

#endif /* GATELAYER_H_ */
