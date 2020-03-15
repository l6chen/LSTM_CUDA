/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <iostream>
#include <string>
#include "gateLayer.h"//actually output gate

#define BLOCK_SIZE 32

namespace gateLayer {

	void GateLayer::init() {
		Whlen = hiddenStates * hiddenStates;
		Wxlen = hiddenStates * embedSize;
		blen = hiddenStates;

		Wh = new float[Whlen];
		Wx = new float[Wxlen];
		b = new float[blen];

		weightbiasTruncInit(Wh, Wx, b, Whlen, Wxlen, blen);

	}

	GateLayer::~GateLayer() {
		delete[] Wh;
		delete[] Wx;
		delete[] b;
		std::cout << "debuginfo: gate is out" << std::endl;
	}

	float* GateLayer::forward(float* h, float* x) {
		
		float* wh = util::matMul(Wh, h, hiddenStates, hiddenStates, 1);
		float* wx = util::matMul(Wx, x, hiddenStates, embedSize, 1);
		float* out = util::matElem(util::matElem(wh, wx, hiddenStates, 1, '+'),
			b, hiddenStates, 1, '+');
		//TODO::make activation a class
		util::sigmoid(out, hiddenStates);

		return out;
	}
	float* GateLayer::backward(float* dh, float* x) {
		return nullptr;
	}

	float* GateLayer::calLoss(float* dh, float* c, float* out) const {

		util::sigmoidPrime(out, hiddenStates);
		util::tanh(c, hiddenStates);
		float* delta = util::matElem(util::matElem(dh, c, 
			hiddenStates, 1, '+'), out, hiddenStates, 1, '+');
		return delta;

	}
	float* GateLayer::calWhGrad(float* hpre, float* delta) const {
		float* hpreT = hpre;//actually same for vector.
		float* Wh_grad = util::matMul(delta, hpreT, hiddenStates, 1, hiddenStates);
		return Wh_grad;
	}

	float* GateLayer::calWxGrad(float* x, float* out) const {
		float* xT = x;
		float* Wx_grad = util::matMul(out, xT, hiddenStates, 1, embedSize);
		return Wx_grad;
	}

	float* GateLayer::calbGrad(float* deltaf) const {
		float* b_grad = deltaf;
		return b_grad;
	}

	void GateLayer::updateWb(float lr, float* Wh_grad, float* Wx_grad, float* b_grad) {
		int h = hiddenStates;
		int e = embedSize;
		Wh = util::matElem(Wh, util::matMulScal(Wh_grad, lr, h, h), h, h, '+');
		Wx = util::matElem(Wh, util::matMulScal(Wx_grad, lr, h, e), h, e, '+');
		b = util::matElem(b, util::matMulScal(b_grad, lr, h, h), h, h, '+');
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
}