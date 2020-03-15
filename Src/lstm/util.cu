/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <vector>
#include <unordered_set>
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

	__global__ void matElemGPU(float* d_A, float* d_B, int nx, int ny, char op) {
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;
		if (ix < nx && iy < ny)
			switch (op)
			{
			case '+':
				d_A[idx] += d_B[idx];
				break;
			case '-':
				d_A[idx] -= d_B[idx];
				break;
			case '*':
				d_A[idx] *= d_B[idx];
				break;
			}
	}

	__global__ void matMulScalGPU(float* d_A, float* d_S, int nx, int ny) {
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;
		if (ix < nx && iy < ny)
			d_A[idx] *= d_S[0];
	}

	__global__ void matMulGPU(float* out, float* d_A, float* d_B, int ny, int nx, int nz)
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

	__global__ void softmaxGPU(float *d_A, int n, const int categories) //problem fixed
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < n) {
			float exp = expf(d_A[idx]);
			d_A[idx] = exp;

			float sum = 0;
			// compute sum
			int startIdx = idx / categories * categories;
			int endIdx = startIdx + categories;
			for (int i = startIdx; i < endIdx; ++i) {
				sum += d_A[i];
			}

			d_A[idx] = d_A[idx] / sum;
		}
	}

	__global__ void tanhActivation(float* d_A, int m)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		float exp1 = expf(d_A[idx]);
		float exp2 = expf(-d_A[idx]);
		if (idx < m) {
			d_A[idx] = (exp1 - exp2) / (exp1 + exp2);
		}
	}

	__global__ void sigmoidActivation(float *d_A, int m)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		float exp = expf(d_A[idx]);
		if(idx < m){
			d_A[idx] = exp / (1.0f + exp);
		}
		
	}

	__global__ void tanhPrimeGPU(float* d_A, int nx) {
		unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < nx)
			d_A[idx] = 1.0f - d_A[idx] * d_A[idx];
	}

	__global__ void sigmoidPrimeGPU(float* d_A, int nx) {
		unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < nx)
			d_A[idx] = (1.0f - d_A[idx]) * d_A[idx];
	}

	__global__ void CrossEntropyLossGPU(float* d_out, float* d_pred,
		float* d_oneHot, int m, int n, bool iftest) //problem
	{
		int nidx = threadIdx.x + blockDim.x * blockIdx.x;
		if (nidx < n) {
			d_out[nidx] = 0;
			for (int midx = 0; midx < m; ++midx) {
				int curIdx = midx * n + nidx;
				if (d_oneHot[curIdx] != 0 && d_pred[curIdx] != 0) {
					d_out[nidx] += -logf(d_pred[curIdx]);
					if (iftest == false)
						break;
				}
			}
		}
	}


	__global__ void TransposeGPU(float *out, float *in, const int nrows, const int ncols)
	{
		int iy = blockIdx.y * blockDim.y + threadIdx.y;
		int ix = blockIdx.x * blockDim.x + threadIdx.x;

		if (iy < nrows && ix < ncols) {
			out[ix*nrows + iy] = in[iy*ncols + ix];
		}
	}

	/*********************************Inplace Matrix Functions*******************************/

	void matElem_inplace(float* matA, float* matB, int m, int n, char op) {
		float* d_A, * d_B;

		//handle supported operation for + - *
		std::unordered_set<char> validOp{'+','-','*'};
		if (validOp.find(op) == validOp.end()) {
			std::cout << "Unsupported operator " << op;
			return;
		}

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_B, m * n * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_A, matA, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, matB, m * n * sizeof(float), cudaMemcpyHostToDevice));

		// invoke kernel
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		matElemGPU << <grid, block >> > (d_A, d_B, n, m, op);

		//transfer data from device to host
		CHECK(cudaMemcpy(matA, d_A, m * n * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_A));
		CHECK(cudaFree(d_B));
	}

	void matTrans_inplace(float* matA, int height, int width) {

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
		float* d_A, * d_out;

		CHECK(cudaMalloc((void**)& d_A, height * width * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_out, height * width * sizeof(float)));
		CHECK(cudaMemcpy(d_A, matA, height * width * sizeof(float), cudaMemcpyHostToDevice));
		TransposeGPU << <grid, block >> > (d_out, d_A, height, width);
		CHECK(cudaMemcpy(matA, d_out, height * width * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(d_A);
		cudaFree(d_out);

	}

	void matMul_inplace(float* out, float* matA, float* matB, int m, int n, int k) {
		float* d_out, * d_A, * d_B;
		CHECK(cudaMalloc((void**)& d_out, m * k * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_B, n * k * sizeof(float)));

		CHECK(cudaMemcpy(d_A, matA, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, matB, n * k * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

		matMulGPU << <grid, block >> > (d_out, d_A, d_B, m, n, k);
		CHECK(cudaMemcpy(out, d_out, m * k * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_out);
		cudaFree(d_A);
		cudaFree(d_B);
	}

	/*********************************Outplace Matrix Functions*******************************/

	float* matElem(const float* matA, const float* matB, int m, int n, char op) {
		float* d_A, * d_B;
		float* out = new float[m * n];
		//handle supported operation for + - *
		std::unordered_set<char> validOp{ '+','-','*' };
		if (validOp.find(op) == validOp.end()) {
			std::cout << "Unsupported operator " << op;
			return nullptr;
		}

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_B, m * n * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_A, matA, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, matB, m * n * sizeof(float), cudaMemcpyHostToDevice));

		// invoke kernel
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		matElemGPU << <grid, block >> > (d_A, d_B, n, m, op);

		//transfer data from device to host
		CHECK(cudaMemcpy(out, d_A, m * n * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_A));
		CHECK(cudaFree(d_B));

		return out;
	}

	float* matMulScal(const float* matA, float scal, int m, int n) {
		float* d_A, * d_S;
		float* out = new float[m * n];

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_S, 1 * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_A, matA, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_S, &scal, 1 * sizeof(float), cudaMemcpyHostToDevice));

		// invoke kernel
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		matMulScalGPU << <grid, block >> > (d_A, d_S, n, m);

		//transfer data from device to host
		CHECK(cudaMemcpy(out, d_A, m * n * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_A));
		CHECK(cudaFree(d_S));

		return out;
	}

	float* matMul(const float* matA, const float* matB, int m, int n, int k) {
		float* out = new float[m * k];
		float* d_out, * d_A, * d_B;
		CHECK(cudaMalloc((void**)& d_out, m * k * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_A, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_B, n * k * sizeof(float)));

		CHECK(cudaMemcpy(d_A, matA, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, matB, n * k * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

		matMulGPU << <grid, block >> > (d_out, d_A, d_B, m, n, k);
		CHECK(cudaMemcpy(out, d_out, m * k * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_out);
		cudaFree(d_A);
		cudaFree(d_B);
		return out;
	}


	float* matTrans(const float* matA, int height, int width) {

		float* out = new float[height * width];
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
		float* d_A, * d_out;

		CHECK(cudaMalloc((void**)& d_A, height * width * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_out, height * width * sizeof(float)));
		CHECK(cudaMemcpy(d_A, matA, height * width * sizeof(float), cudaMemcpyHostToDevice));
		TransposeGPU << <grid, block >> > (d_out, d_A, height, width);
		CHECK(cudaMemcpy(out, d_out, height * width * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(d_A);
		cudaFree(d_out);

		return out;
	}


	/*********************************Activation Functions*******************************/
	void softmax(float* A, int n, const int categories){
		float *d_A;
		CHECK(cudaMalloc((void**)&d_A, n * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x);
		softmaxGPU << <grid, block >> > (d_A, n, categories);
		CHECK(cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);
	
	}
	
	void tanh(float* A, int n){

		float *d_A;
		CHECK(cudaMalloc((void**)&d_A, n * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x);
		tanhActivation << <grid, block >> > (d_A, n);
		CHECK(cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);
	
	}
	
	
	void sigmoid(float *A, int n){
		
		float *d_A;
		CHECK(cudaMalloc((void**)&d_A, n * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x);
		sigmoidActivation << <grid, block >> > (d_A, n);
		CHECK(cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);
	}

	void sigmoidPrime(float *A, int n) {
		
		float* d_A;
		CHECK(cudaMalloc((void**)& d_A, n * sizeof(float)));
		CHECK(cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x);

		sigmoidPrimeGPU << <grid, block >> > (d_A, n);
		CHECK(cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);

	}


	void tanhPrime(float* matA, int n) {
		float* d_A;
		CHECK(cudaMalloc((void**)& d_A, n * sizeof(float)));
		CHECK(cudaMemcpy(d_A, matA, n * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x);

		tanhPrimeGPU << <grid, block >> > (d_A, n);
		CHECK(cudaMemcpy(matA, d_A, n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_A);
	}

	float crossEntropyLoss(float* pred, float* oneHot, int m, int n, bool iftest) { 
		float* d_pred, *d_oneHot, *d_out;
		float* out = (float*)malloc(m * sizeof(float));
		CHECK(cudaMalloc((void**)& d_pred, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_oneHot, m * n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_out, m * sizeof(float)));
		CHECK(cudaMemcpy(d_pred, pred, m * n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_oneHot, oneHot, m * n * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x);

		CrossEntropyLossGPU << <grid, block >> > (d_out, d_pred, d_oneHot, m, n, iftest);
		CHECK(cudaMemcpy(out, d_out, m * sizeof(float), cudaMemcpyDeviceToHost));

		float avgcost = 0;
		for (int i = 0; i < m; ++i) {
			avgcost += out[i];
		}
		avgcost /= m;
		cudaFree(d_pred);
		cudaFree(d_oneHot);
		cudaFree(d_out);
		return avgcost;
	}








}