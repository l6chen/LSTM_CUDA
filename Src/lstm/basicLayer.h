/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef BASICLAYER_H_
#define BASICLAYER_H_

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
	struct OutputsDelta {
		std::vector<float*> fgs, igs, ogs, cts, cgs, hs;
		std::vector<float*> dfs, dis, dos, dcts, dhs;
	};
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
		virtual float* forward(float* x, float* h, void (*activate)(float* A, int n)) { return nullptr; }
		virtual float* backward(float* dh, float* x) { return nullptr; }
		virtual void init() {}
		float* concatVec(float* vecA, float* vecB, const int a, const int b);
		void randInit();
		void showVar() const;
		void weightbiasTruncInit(float* Wh, float* Wx, float* b, 
			const int Whlen, const int Wxlen, const int blen);
		void weightbiasGradInit(float* Wh, float* Wx, float* b,
			const int Whlen, const int Wxlen, const int blen);
		void denseweightbiasTruncInit(float* W, float* b,
			const int Wlen, const int blen);
		void denseweightbiasGradInit(float* W, float* b,
			const int Wlen, const int blen);

	};
}

#endif /* BASICLAYER_H_ */
