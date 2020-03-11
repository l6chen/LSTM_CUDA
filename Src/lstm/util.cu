/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <vector>
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "util.h"
#include "../common/common.h"
#define BLOCK_SIZE 32

namespace util {

	__global__ void matrixSumGPU(float* d_A, float* d_B, int nx, int ny) {
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;
		if (ix < nx && iy < ny)
			d_A[idx] += d_B[idx];
	}

	__global__ void matrixMulElemGPU(float* d_A, float* d_B, int nx, int ny) {
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;
		if (ix < nx && iy < ny)
			d_A[idx] *= d_B[idx];
	}

	__global__ void matrixMulGPU(float* out, float* d_A, float* d_B, int ny, int nx, int nz)
	{
		int iy = blockIdx.y * blockDim.y + threadIdx.y;
		int iz = blockIdx.x * blockDim.x + threadIdx.x;
		float sum = 0;
		if (iz < nz && iy < ny)
		{
			for (int ix = 0; ix < nx; ix++)
			{
				sum += d_A[iy * nx + ix] * d_B[ix * nz + iz];
			}
			out[iy * nz + iz] = sum;
		}
	}

	void matrixSum(float* matA, float* matB, int m, int n) {
		float* d_A, *d_B;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_B, m * n * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_A, matA, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, matB, m * n * sizeof(float), cudaMemcpyHostToDevice));

		// invoke kernel
		dim3 block(BLOCK_SIZE,1);
		dim3 grid((n + block.x - 1) / block.x, m);
		matrixSumGPU << <grid, block >> > (d_A, d_B, n, m);

		//transfer data from device to host
		CHECK(cudaMemcpy(matA, d_A, m * n * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_A));
		CHECK(cudaFree(d_B));
	}

	void matrixMulElem(float* matA, float* matB, int m, int n) {
		float* d_A, * d_B;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_B, m * n * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_A, matA, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, matB, m * n * sizeof(float), cudaMemcpyHostToDevice));

		// invoke kernel
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		matrixMulElemGPU << <grid, block >> > (d_A, d_B, n, m);

		//transfer data from device to host
		CHECK(cudaMemcpy(matA, d_A, m * n * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_A));
		CHECK(cudaFree(d_B));
	}

	void matrixMul(float* out, float* matA, float* matB, int m, int n, int k) {
		float* d_out, * d_A, * d_B;
		CHECK(cudaMalloc((void**)& d_out, m * k * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_B, n * k * sizeof(float)));

		CHECK(cudaMemcpy(d_A, matA, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, matB, n * k * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

		matrixMulGPU << <grid, block >> > (d_out, d_A, d_B, m, n, k);
		CHECK(cudaMemcpy(out, d_out, m * k * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_out);
		cudaFree(d_A);
		cudaFree(d_B);
	}

	void softmax(){}
	void tanh(){}
	void sigmoid(){}
}