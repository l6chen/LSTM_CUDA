/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef TESTLAYER_H_
#define TESTLAYER_H_

#include <vector>
#include <iostream>
#include "../lstm/basicLayer.h"
#include "../lstm/gateLayer.h"
#include "../lstm/denseLayer.h"
#include "../lstm/embedLayer.h"

namespace testLayer {
	void testBasicLayer();
	void testGateLayer();
	void testDenseLayer();
	void testEmbedLayer();
}


#endif