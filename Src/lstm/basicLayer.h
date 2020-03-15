/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef BASICLayer_H_
#define BASICLayer_H_

#include <vector>
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <string>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

namespace basicLayer {
	class BasicLayer {
	protected:
		const int embedSize;//
		const int timeStep;//updatetimes
		const int hiddenStates;//m 
		const int categories;
		curandState* rndstates = NULL;//device variable
		void showConcat(float* vec, const int len) const;
		int allVar[4] = { embedSize, timeStep, hiddenStates, categories };

	public:
		BasicLayer(int embeds, int times, int hid, int cat) :
					embedSize(embeds), timeStep(times),
			        hiddenStates(hid), categories(cat) {}
		virtual ~BasicLayer() {}
		virtual float* forward(float* h, float* x) { return nullptr; }
		virtual float* backward(float* dh, float* x) { return nullptr; }
		virtual void init() {}
		float* concatVec(float* vecA, float* vecB, const int a, const int b);
		void randInit();
		void showVar() const;
		void weightbiasTruncInit(float* Wh, float* Wx, float* b, 
			const int Whlen, const int Wxlen, const int blen);
	};
}

#endif /* BASICLayer_H_ */
