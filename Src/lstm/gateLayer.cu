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

	GateLayer::GateLayer(int embeds, int times, int hid, int cat, float lr) :
		basicLayer::BasicLayer(embeds, times, hid, cat, lr) {
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

	void GateLayer::calGrad(float* x, basicLayer::OutputsDelta* datas, std::vector<float*>* dgates)
	{
		WbGradInit();//for each x, necessary to reinitialize grad to zero.
		auto hs = datas->hs;
		auto dfs = datas->dfs;
		for (int k = curTime; k > 0; k--) {
			//WhGrad,dgates[0] is for unpoint/unref
			util::matElem_inplace(WhGrad, util::matMul(dgates[0][k], hs[k - 1],
				hiddenStates, 1, hiddenStates), hiddenStates, hiddenStates, '+');
			//bGrad
			util::matElem_inplace(bGrad, dfs[k], hiddenStates, 1, '+');
		}
		//WxGrad Wrong HERE
		WxGrad = util::matMul(dgates->back(), x, hiddenStates, 1, embedSize);

	}

	void GateLayer::updateWb() {
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
	void OutputGate::calDeltak(basicLayer::OutputsDelta* datas, int k) 
	{
		auto og = datas->ogs[k];
		auto dh = datas->dhs[k];
		auto cg = datas->cgs[k];
		
		util::tanh(cg, hiddenStates);
		util::sigmoidPrime(og, hiddenStates);
		datas->dos[k] = util::matElem(util::matElem(dh,
			cg, hiddenStates, 1, '*'),
			og, hiddenStates, 1, '*');


	}

	/****************************InputLayer Implementation***************************/

	void InputGate::calDeltak(basicLayer::OutputsDelta* datas, int k)
	{
		auto ig = datas->igs[k];
		auto og = datas->ogs[k];
		auto dh = datas->dhs[k];
		auto cg = datas->cgs[k];
		auto ct = datas->cts[k];


		util::tanh(cg, hiddenStates);
		util::tanhPrime(cg, hiddenStates);
		util::sigmoidPrime(ig, hiddenStates);
		datas->dis[k] = util::matElem(util::matElem(util::matElem(util::matElem(dh,
			og, hiddenStates, 1, '*'),
			cg, hiddenStates, 1, '*'),
			ct, hiddenStates, 1, '*'),
			ig, hiddenStates, 1, '*');


	}

	/****************************ForgetLayer Implementation***************************/

	void ForgetGate::calDeltak(basicLayer::OutputsDelta* datas, int k)
	{
		auto fg = datas->fgs[k];
		auto og = datas->ogs[k];
		auto dh = datas->dhs[k];
		auto cg = datas->cgs[k];
		auto cp = datas->cgs[k - 1];
		auto ct = datas->cts[k];


		util::tanh(cg, hiddenStates);
		util::tanhPrime(cg, hiddenStates);
		util::sigmoidPrime(fg, hiddenStates);
		datas->dfs[k] = util::matElem(util::matElem(util::matElem(util::matElem(dh,
			og, hiddenStates, 1, '*'),
			cg, hiddenStates, 1, '*'),
			cp, hiddenStates, 1, '*'),
			fg, hiddenStates, 1, '*');


	}

	/****************************CellTempLayer Implementation***************************/

	void CellTGate::calDeltak(basicLayer::OutputsDelta* datas, int k)
	{
		auto ig = datas->fgs[k];
		auto og = datas->ogs[k];
		auto dh = datas->dhs[k];
		auto cg = datas->cgs[k];
		auto ct = datas->cts[k];


		util::tanh(cg, hiddenStates);
		util::tanhPrime(cg, hiddenStates);
		util::tanhPrime(ct, hiddenStates);
		datas->dcs[k] = util::matElem(util::matElem(util::matElem(util::matElem(dh,
			og, hiddenStates, 1, '*'),
			cg, hiddenStates, 1, '*'),
			ig, hiddenStates, 1, '*'),
			ct, hiddenStates, 1, '*');


	}

	/****************************Special Layers Implementation***************************/

	float* CellGate::forward(basicLayer::OutputsDelta* datas) {
		float* fg = datas->fgs[curTime];
		float* cgprev = datas->cgs[curTime - 1];
		float* ig = datas->igs[curTime];
		float* ct = datas->cts[curTime];

		float* fc = util::matElem(fg, cgprev, hiddenStates, hiddenStates, '*');
		float* ic = util::matElem(ig, ct, hiddenStates, hiddenStates, '*');
		return util::matElem(fc, ic, hiddenStates, 1, '+');
	}


	float* HiddenGate::forward(basicLayer::OutputsDelta* datas) {
		float* og = datas->ogs[curTime];
		float* ct = datas->cts[curTime];

		util::tanh(ct, hiddenStates);
		return util::matElem(og, ct, hiddenStates, 1, '*');
	}
}