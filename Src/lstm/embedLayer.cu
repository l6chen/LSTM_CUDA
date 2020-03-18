/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <iostream>
#include <string>
#include "embedLayer.h"

#define BLOCK_SIZE 32


namespace embedLayer {

	/****************************EmbedLayer Implementation***************************/

	EmbedLayer::EmbedLayer(int embeds, int times, int hid, int cat, int dic) :
		basicLayer::BasicLayer(embeds, times, hid, cat),  dictSize(dic){
		Wlen = embeds * dic;

		W = new float[Wlen];

		init();
	}

	EmbedLayer::~EmbedLayer() {
		delete[] W;
		std::cout << "debuginfo: embedLayer is out" << std::endl;
	}

	void EmbedLayer::init() {
		embedweightTruncInit(W, Wlen);
	}

	inline void EmbedLayer::WbGradInit() {
		embedweightGradInit(W, Wlen);
	}

	float* EmbedLayer::forward(int textCode)
	{
		float* out = W + embedSize * textCode;

		return out;
	}

	void EmbedLayer::calGrad(float* delta, int textCode)
	{
		WbGradInit();//for each h, necessary to reinitialize grad to zero.
		
		for (int iy = 0; iy < embedSize; iy++) {
			WGrad[iy * dictSize + textCode] = delta[iy];
		}

	}

	void EmbedLayer::updateWb(float lr) {
		int d = dictSize;
		int e = embedSize;
		W = util::matElem(W, util::matMulScal(WGrad, lr, e, d), e, d, '-');
	}

	//Showing information
	void EmbedLayer::showW() const {
		for (int i = 0; i < Wlen; i++)
			std::cout << W[i] << " ";
		std::cout << std::endl;
	}


	void EmbedLayer::showforward(float* out) const {
		int outlen = categories;
		for (int i = 0; i < outlen; i++)
			std::cout << out[i] << " ";
		std::cout << std::endl;
	}



}