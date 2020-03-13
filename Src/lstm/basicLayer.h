/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef BASICLayer_H_
#define BASICLayer_H_

#include <vector>
#include <iostream>

namespace basicLayer {
	class BasicLayer {
	protected:
		const int embedSize;//
		const int timeStep;//updatetimes
		const int hiddenStates;//m 
		const int categories;
		int allVar[4] = { embedSize, timeStep, hiddenStates, categories };

	public:
		BasicLayer(int embeds, int times, int hid, int cat) :
					embedSize(embeds), timeStep(times),
			        hiddenStates(hid), categories(cat) {}
		virtual ~BasicLayer() {}
		virtual float* forward(float* prevMat) { return nullptr; }
		virtual float* backward(float* gradMat) { return nullptr; }
		virtual void init() {}
		float* concatVec(float* vecA, float* vecB, const int a, const int b);
		void showVar() const;
	};
}

#endif /* BASICLayer_H_ */
