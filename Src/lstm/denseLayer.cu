/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <iostream>
#include <string>
#include "denseLayer.h"

#define BLOCK_SIZE 32


namespace denseLayer {

	/****************************DenseLayer Implementation***************************/

	DenseLayer::DenseLayer(int embeds, int times, int hid, int cat, float lr) :
		basicLayer::BasicLayer(embeds, times, hid, cat, lr) {
		Wlen = categories * hiddenStates;
		blen = categories;

		W = new float[Wlen];
		b = new float[blen];

		init();
	}

	DenseLayer::~DenseLayer() {
		delete[] W;
		delete[] b;
		std::cout << "debuginfo: denseLayer is out" << std::endl;
	}

	void DenseLayer::init() {
		denseweightbiasTruncInit(W, b, Wlen, blen);
	}

	inline void DenseLayer::WbGradInit() {
		denseweightbiasGradInit(WGrad, bGrad, Wlen, blen);
	}

	float* DenseLayer::forward(float* x, float* h, void (*activate)(float* A, int n))
	{
		float* wh = util::matMul(W, h, categories, hiddenStates, 1);
		float* out = util::matElem(wh, b, categories, 1, '+');
		(*activate)(out, categories);
		return out;
	}

	void DenseLayer::calGrad(float* delta, float* h)
	{
		WbGradInit();//for each h, necessary to reinitialize grad to zero.
		
		//y is pred, t is true, h is input of dense, output of LSTM
		WGrad = util::matMul(delta, h, categories, 1, hiddenStates);
		bGrad = delta;

	}

	void DenseLayer::updateWb(float lr) {
		int h = hiddenStates;
		W = util::matElem(W, util::matMulScal(WGrad, lr, categories, h), categories, h, '-');
		b = util::matElem(b, util::matMulScal(bGrad, lr, categories, 1), categories, 1, '-');
	}

	//Showing information
	void DenseLayer::showW() const {
		for (int i = 0; i < Wlen; i++)
			std::cout << W[i] << " ";
		std::cout << std::endl;
	}

	void DenseLayer::showb() const {
		for (int i = 0; i < blen; i++)
			std::cout << b[i] << " ";
		std::cout << std::endl;
	}

	void DenseLayer::showforward(float* out) const {
		int outlen = categories;
		for (int i = 0; i < outlen; i++)
			std::cout << out[i] << " ";
		std::cout << std::endl;
	}

	float DenseLayer::calLoss(float* p, float* t) {
		
		return util::crossEntropyLoss(p, t, categories, 1);
	}

	float* DenseLayer::backward(float* h, float* y, float* t) {
		float* delta = util::matElem(y, t, categories, 1, '-');
		float* deltah = util::matMul(util::matTrans(W,
			categories, hiddenStates), delta, hiddenStates, categories, 1);//may need 1-y^2
		calGrad(delta, h);
		updateWb(lr);
		return deltah;
	}
}