/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include "basicLayer.h"
#define BLOCK_SIZE 32

namespace basicLayer {

	__global__ void randInitGPU(int nx, int ny, curandState* rndstates) {
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;
		if (ix < nx && iy < ny) {
			curand_init(clock() + idx, idx, 0, &rndstates[idx]);
		}
			
	}

	__global__ void weightbiasTruncInitGPU(int nx, int ny, float* d_W,
		float* d_b, curandState* rndstates) { // truncated uniform
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;

		//for Win
		if (ix < nx && iy < ny - 1) {
			d_W[idx] = curand_uniform(&rndstates[idx]) * 2.0E-4 - 1.0E-4;
		}//for bin
		else if (ix < nx && iy == ny - 1) {
			d_b[ix] = curand_uniform(&rndstates[idx]) * 2.0E-4 - 1.0E-4;
		}

	}
	__global__ void concatVecGPU(const int nx, const int ny, float* d_out, 
		float* d_A, float* d_B){
		unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < nx) {
			d_out[idx] = d_A[idx];
		}
		else if (idx < nx + ny) {
			d_out[idx] = d_B[idx - nx];
		}
	
	}

	void BasicLayer::randInit() {
		int m = hiddenStates + 1;
		int n = embedSize + hiddenStates;
		cudaMalloc((void**)& rndstates, m * n * sizeof(curandState));
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		randInitGPU << <grid,block >> > (n, m, rndstates);
	}

	void BasicLayer::weightbiasTruncInit(float* W, float* b, const int Wlen, const int blen) {
		randInit();

		float* d_W, * d_b;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_W, Wlen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_b, blen * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_W, W, Wlen * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_b, b, blen * sizeof(float), cudaMemcpyHostToDevice));

		//randomlize weights and bias
		int m = hiddenStates + 1;
		int n = blen;

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		weightbiasTruncInitGPU << <grid, block >> > (n, m, d_W, d_b, rndstates);

		//transfer data from device to host
		CHECK(cudaMemcpy(W, d_W, Wlen * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(b, d_b, blen * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_W));
		CHECK(cudaFree(d_b));

	}

	float* BasicLayer::concatVec(float* vecA, float* vecB,
		const int alen, const int blen) {
		const int outlen = alen + blen;
		float* out = new float[outlen];
		
		float* d_A, * d_B, * d_out;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_A, alen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_B, blen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_out, outlen * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_A, vecA, alen * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, vecB, blen * sizeof(float), cudaMemcpyHostToDevice));

		//Invoke kernel
		dim3 block(BLOCK_SIZE);
		dim3 grid((outlen + block.x - 1) / block.x);
		concatVecGPU << <grid, block >> > (alen, blen, d_out, d_A, d_B);

		//transfer data from device to host
		CHECK(cudaMemcpy(out, d_out, outlen * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_A));
		CHECK(cudaFree(d_B));
		CHECK(cudaFree(d_out));

		return out;

	}
	void BasicLayer::showVar() const {
		for (int i = 0; i < 4; i++) {
			std::cout << allVar[i];
		}
		std::cout << std::endl;
	}
	void BasicLayer::showConcat(float* vec, const int len) const {
		for (int i = 0; i < len; i++) {
			std::cout << vec[i] << " ";
		}
		std::cout << std::endl;
	}
}