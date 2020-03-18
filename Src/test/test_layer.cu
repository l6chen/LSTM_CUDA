/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/


#include <vector>
#include <iostream>
#include "test_Layer.h"


namespace testLayer {
	void testBasicLayer() {
		basicLayer::BasicLayer* bd = new basicLayer::BasicLayer(0, 2, 4, 5);
		bd->showVar();
		float* a = new float[3];
		float* b = new float[4];
		for (int i = 0; i < 3; i++) {
			a[i] = 0.1f;
		}
		for (int i = 0; i < 4; i++) {
			b[i] = 0.2f;
		}
		float* out = bd->concatVec(a, b, 3, 4);
		for (int i = 0; i < 7; i++) {
			std::cout << out[i] << " ";
		}
		delete[] a;
		delete[] b;
		delete bd;
		std::cout << "\n\n";
	}
	void testGateLayer() {
		gateLayer::GateLayer* ig = new gateLayer::GateLayer(5, 5, 10, 3);
		ig->showVar();
		ig->showW();
		ig->showb();

		delete ig;
		std::cout << "\n\n";
	}

	void testDenseLayer() {
		denseLayer::DenseLayer* dl = new denseLayer::DenseLayer(5, 5, 10, 3);
		dl->showVar();
		dl->showW();
		dl->showb();

		delete dl;
		std::cout << "\n\n";
	}

	void testEmbedLayer() {
		embedLayer::EmbedLayer* el = new embedLayer::EmbedLayer(10, 5, 10, 3, 18);

		el->showW();


		delete el;
		std::cout << "\n\n";
	}
}