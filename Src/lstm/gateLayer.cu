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

	void GateLayer::init() {
		int Wlen = hiddenStates * (hiddenStates + embedSize);
		int blen = hiddenStates + embedSize;

		Win = new float[Wlen];
		bin = new float[blen];

		weightbiasTruncInit(Win, bin, Wlen, blen);

		std::cout << "debuginfo: gateLayer initiated." << std::endl;
	}

	float* GateLayer::forward(float* h, float* x) {
		float* hx = concatVec(h, x, hiddenStates, embedSize);
		int inlen = hiddenStates;
		float* in = new float[inlen];
		util::matrixMul(in, Win, hx, hiddenStates, hiddenStates + embedSize, 1);
		free(hx);
		util::matrixCalElem(in, bin, 1, hiddenStates + embedSize, '+');
		util::sigmoid(in, hiddenStates + embedSize);
		return in;
	}
	float* GateLayer::backward(float* dh, float* x) {
		return nullptr;
	}
	void GateLayer::updateWb() {}
	void GateLayer::calLoss(float* dh, float* x) const {}
	void GateLayer::calGrad(float* dh, float* x) const {}

	//Showing information
	void GateLayer::showW() const {
		int Wlen = hiddenStates * (hiddenStates + embedSize);
		for (int i = 0; i < Wlen; i++)
			std::cout << Win[i] << " ";
		std::cout << std::endl;
	}

	void GateLayer::showb() const {
		int blen = hiddenStates + embedSize;
		for (int i = 0; i < blen; i++)
			std::cout << bin[i] << " ";
		std::cout << std::endl;
	}

	void GateLayer::showforward(float* in) const {
		int inlen = hiddenStates + embedSize;
		for (int i = 0; i < inlen; i++)
			std::cout << in[i] << " ";
		std::cout << std::endl;
	}
}