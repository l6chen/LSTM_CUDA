/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include "lstm.h"

namespace lstm {
	LSTMNetwork::LSTMNetwork(float lr, int rnnSize, int numLSTM) {
		_lr = lr;
		_rnnSize = rnnSize;
		_numLSTM = numLSTM;
	}
	void LSTMNetwork::train(){}
}