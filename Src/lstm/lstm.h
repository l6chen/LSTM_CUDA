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

namespace lstm {
	class LSTMNetwork {
	private:
		float _lr;
		int _rnnSize;
		int _numLSTM;
		int curStep = 0;
		int embedSize = 10;
		basicLayer::OutputsDelta datas;
	public:
		LSTMNetwork(float lr, int rnnSize = 512, int numLSTM = 1);
		void train();
	};
}
#endif /* LSTM_H_ */