/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef LSTM_H_
#define LSTM_H_

#include <vector>
#include <iostream>
#include "basicLayer.h"
#include "gateLayer.h"
#include "dataLoader.h"
#include "util.h"
#include "denseLayer.h"
#include "embedLayer.h"

namespace lstm {
	class LSTMNetwork {
	private:
		float lrate;
		int categories;
		int hiddenStates;
		int embedSize;
		int timeSteps = -1;
		int epoch;
		int dictSize;
		basicLayer::OutputsDelta* oD;
		
		int curStep = 0;

		float* forward(std::unordered_map<std::string, basicLayer::BasicLayer*>&
			layers, std::vector<int> x);
		float backward(float* pred, float* t, std::vector<int> x,
			std::unordered_map<std::string, basicLayer::BasicLayer*>& layers);
		std::vector<float*> getDeltaEmbed(float* Wfx, float* Wix,
			float* Wcx, float* Wox);
		
		void OutputsDeltaInit();

	public:
		LSTMNetwork(float lr, int cat, int hid, int emb, int ep);
		void train(dataLoader::DataSets* ds);
		void test();
		
	};
}
#endif /* LSTM_H_ */