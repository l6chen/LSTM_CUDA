/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <iostream>
#include <string>
#include "gateLayer.h"

#define BLOCK_SIZE 32

namespace gateLayer {

	/****************************GateLayer Implementation***************************/

	GateLayer::GateLayer(int embeds, int times, int hid, int cat) :
		basicLayer::BasicLayer(embeds, times, hid, cat) {
		Whlen = hiddenStates * hiddenStates;
		Wxlen = hiddenStates * embedSize;
		blen = hiddenStates;

		Wh = new float[Whlen];
		Wx = new float[Wxlen];
		b = new float[blen];

		init();
	}

	GateLayer::~GateLayer() {
		delete[] Wh;
		delete[] Wx;
		delete[] b;
		std::cout << "debuginfo: gateLayer is out" << std::endl;
	}

	void GateLayer::init() { weightbiasTruncInit(Wh, Wx, b, Whlen, Wxlen, blen);}

	inline void GateLayer::WbGradInit() { 
		weightbiasGradInit(WhGrad, WxGrad, bGrad, Whlen, Wxlen, blen);
	}

	float* GateLayer::forward(float* x, float* h, void (*activate)(float* A, int n)) 
	{
		float* wh = util::matMul(Wh, h, hiddenStates, hiddenStates, 1);
		float* wx = util::matMul(Wx, x, hiddenStates, embedSize, 1);
		float* out = util::matElem(util::matElem(wh, wx, hiddenStates, 1, '+'),
			b, hiddenStates, 1, '+');
		(*activate)(out, hiddenStates);
		curTime++;
		return out;
	}

	void GateLayer::calGrad(float* x, basicLayer::OutputsDelta& datas, std::vector<float*> dgates)
	{
		WbGradInit();//for each x, necessary to reinitialize grad to zero.
		auto hs = datas.hs;
		auto dfs = datas.dfs;
		for (int k = curTime; k > 0; k--) {
			//WhGrad
			util::matElem_inplace(WhGrad, util::matMul(dgates[k], hs[k - 1],
				hiddenStates, 1, hiddenStates), hiddenStates, hiddenStates, '+');
			//bGrad
			util::matElem_inplace(bGrad, dfs[k], hiddenStates, 1, '+');
		}
		//WxGrad
		WxGrad = util::matMul(dgates.back(), x, hiddenStates, 1, embedSize);

	}

	void GateLayer::updateWb(float lr) {
		int h = hiddenStates;
		int e = embedSize;
		Wh = util::matElem(Wh, util::matMulScal(WhGrad, lr, h, h), h, h, '-');
		Wx = util::matElem(Wh, util::matMulScal(WxGrad, lr, h, e), h, e, '-');
		b = util::matElem(b, util::matMulScal(bGrad, lr, h, h), h, h, '-');
	}

	//Showing information
	void GateLayer::showW() const {
		for (int i = 0; i < Whlen; i++)
			std::cout << Wh[i] << " ";
		std::cout << std::endl;
		for (int i = 0; i < Wxlen; i++)
			std::cout << Wx[i] << " ";
		std::cout << std::endl;
	}

	void GateLayer::showb() const {
		for (int i = 0; i < blen; i++)
			std::cout << b[i] << " ";
		std::cout << std::endl;
	}

	void GateLayer::showforward(float* out) const {
		int outlen = hiddenStates;
		for (int i = 0; i < outlen; i++)
			std::cout << out[i] << " ";
		std::cout << std::endl;
	}

	/****************************OutputLayer Implementation***************************/

	//TODO self.delta_h_list[-1] = delta_h
	void OutputGate::calDelta(float* x, basicLayer::OutputsDelta& datas) 
	{
		auto ogs = datas.ogs;
		auto dhs = datas.dhs;
		auto cgs = datas.cgs;
		
		for (int k = curTime; k > 0; k--) {
			util::tanh(cgs[k], hiddenStates);
			util::sigmoidPrime(ogs[k], hiddenStates);
			datas.dos[k] = util::matElem(util::matElem(dhs[k],
				cgs[k], hiddenStates, 1, '*'),
				ogs[k], hiddenStates, 1, '*');
		}

	}

	/****************************InputLayer Implementation***************************/

	void InputGate::calDelta(float* x, basicLayer::OutputsDelta& datas)
	{
		auto igs = datas.igs;
		auto ogs = datas.ogs;
		auto dhs = datas.dhs;
		auto cgs = datas.cgs;
		auto cts = datas.cts;

		for (int k = curTime; k > 0; k--) {
			util::tanh(cgs[k], hiddenStates);
			util::tanhPrime(cgs[k], hiddenStates);
			util::sigmoidPrime(igs[k], hiddenStates);
			datas.ogs[k] = util::matElem(util::matElem(util::matElem(util::matElem(dhs[k],
				ogs[k], hiddenStates, 1, '*'),
				cgs[k], hiddenStates, 1, '*'),
				cts[k], hiddenStates, 1, '*'),
				igs[k], hiddenStates, 1, '*');
		}

	}

	/****************************ForgetLayer Implementation***************************/

	void ForgetGate::calDelta(float* x, basicLayer::OutputsDelta& datas)
	{
		auto fgs = datas.fgs;
		auto ogs = datas.ogs;
		auto dhs = datas.dhs;
		auto cgs = datas.cgs;
		auto cts = datas.cts;

		for (int k = curTime; k > 0; k--) {
			util::tanh(cgs[k], hiddenStates);
			util::tanhPrime(cgs[k], hiddenStates);
			util::sigmoidPrime(fgs[k], hiddenStates);
			datas.ogs[k] = util::matElem(util::matElem(util::matElem(util::matElem(dhs[k],
				ogs[k], hiddenStates, 1, '*'),
				cgs[k], hiddenStates, 1, '*'),
				cgs[k - 1], hiddenStates, 1, '*'),
				fgs[k], hiddenStates, 1, '*');
		}

	}

	/****************************CellTempLayer Implementation***************************/

	void CellTGate::calDelta(float* x, basicLayer::OutputsDelta& datas)
	{
		auto igs = datas.fgs;
		auto ogs = datas.ogs;
		auto dhs = datas.dhs;
		auto cgs = datas.cgs;
		auto cts = datas.cts;

		for (int k = curTime; k > 0; k--) {
			util::tanh(cgs[k], hiddenStates);
			util::tanhPrime(cgs[k], hiddenStates);
			util::tanhPrime(cts[k], hiddenStates);
			datas.ogs[k] = util::matElem(util::matElem(util::matElem(util::matElem(dhs[k],
				ogs[k], hiddenStates, 1, '*'),
				cgs[k], hiddenStates, 1, '*'),
				igs[k], hiddenStates, 1, '*'),
				cts[k], hiddenStates, 1, '*');
		}

	}

	/****************************Special Layers Implementation***************************/

	float* CellGate::forward(basicLayer::OutputsDelta datas) {
		auto fg = datas.fgs[curTime];
		auto cgprev = datas.cgs[curTime - 1];
		auto ig = datas.igs[curTime];
		auto ct = datas.cts[curTime];

		float* fc = util::matElem(fg, cgprev, hiddenStates, hiddenStates, '*');
		float* ic = util::matElem(ig, ct, hiddenStates, hiddenStates, '*');
		return util::matElem(fc, ic, hiddenStates, 1, '+');
	}


	float* HiddenGate::forward(basicLayer::OutputsDelta datas) {
		auto og = datas.ogs[curTime];
		auto ct = datas.cts[curTime];

		util::tanh(ct, hiddenStates);
		return util::matElem(og, ct, hiddenStates, 1, '*');
	}
}