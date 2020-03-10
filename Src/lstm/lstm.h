/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef LSTM_H_
#define LSTM_H_

#include <vector>
#include <iostream>

namespace lstm {
	class LSTMNetwork {
	private:
		float _lr;
		int _rnnSize;
		int _numLSTM;

	public:
		LSTMNetwork(float lr, int rnnSize = 512, int numLSTM = 1);
	};
}
#endif /* LSTM_H_ */