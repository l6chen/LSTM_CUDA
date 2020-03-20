/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include "basicLayer.h"
#define BLOCK_SIZE 128

namespace basicLayer {

	__global__ void randInitGPU(int nx, int ny, curandState* rndstates) {
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;
		if (ix < nx && iy < ny) {
			curand_init(clock() + idx, idx, 0, &rndstates[idx]);
		}
			
	}

	__global__ void weightbiasTruncInitGPU(int hid, int emb, float* d_Wh, float* d_Wx,
		float* d_b, curandState* rndstates) { // truncated uniform
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idh = iy * hid + emb + ix;

		//for Wh
		if (ix < hid && iy < hid) {
			d_Wh[iy * hid + ix] = curand_uniform(&rndstates[idh]) * 2.0E-3 - 1.0E-3;
		}//for Wx
		else if (ix < hid && iy < hid + emb) {
			d_Wx[(iy - hid) * hid + ix] = curand_uniform(&rndstates[idh]) * 2.0E-3 - 1.0E-3;
		}//for bin
		else if (ix < hid && iy == hid + emb) {
			d_b[ix] = 0.0f;
		}
		        
	}

	__global__ void weightbiasGradInitGPU(int hid, int emb, float* d_Wh, float* d_Wx,
		float* d_b) { 
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idh = iy * (hid + emb) + ix;

		//for Wh
		if (ix < hid && iy < hid) {
			d_Wh[iy * hid + ix] = 0.0f;
		}//for Wx
		else if (ix < hid && iy < hid + emb) {
			d_Wx[(iy - hid) * hid + ix] = 0.0f;
		}//for bin
		else if (ix < hid && iy == hid + emb) {
			d_b[ix] = 0.0f;
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

	__global__ void denseweightbiasTruncInitGPU(int nx, int ny, float* d_W,
		float* d_b, curandState* rndstates, int layerId) {// truncated uniform
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;

		if (layerId == 1) {//for dense
			//for W
			if (ix < nx - 1 && iy < ny) {
				d_W[iy * (nx - 1) + ix] = curand_uniform(&rndstates[idx]) * 2.0E-3 - 1.0E-3;
			}//for b
			else if (ix == nx - 1 && iy < ny) {
				d_b[iy] = 0.0f;
			}
		}
		else {//for embed
			if (ix < nx && iy < ny)
				d_W[idx] = curand_uniform(&rndstates[idx]) * 2.0E-3 - 1.0E-3;
		}
	}

	__global__ void embedweightbiasTruncInitGPU(int nx, int ny, float* d_W,
		curandState* rndstates) {// truncated uniform
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;

		if (ix < nx && iy < ny)
			d_W[idx] = curand_uniform(&rndstates[idx]) * 2.0E-3 - 1.0E-3;
	
	}

	__global__ void denseweightbiasGradInitGPU(int nx, int ny, float* d_W,
		float* d_b, curandState* rndstates, int layerId) {
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int iy = blockIdx.y;
		unsigned int idx = iy * nx + ix;

		if (layerId == 1) {//for dense
		//for W
			if (ix < nx - 1 && iy < ny) {
				d_W[iy * (nx - 1) + ix] = 0.0f;
			}//for b
			else if (ix == nx - 1 && iy < ny) {
				d_b[iy] = 0.0f;
			}
		}
		else {//for dense
			if (ix < nx && iy < ny)
				d_W[idx] = 0.0f;
		}
	}

	__global__ void embedCalGradGPU(float* W, float* delta, int textCode, int nx, int ny) {
		unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

		if (ix < nx) {
			W[ix * ny + textCode] = delta[ix];
		}
	}


	float* BasicLayer::embedCalGrad(float* W, float* delta, int textCode, int nx, int ny) {
		int n = nx * ny;
		float* out = new float[n];
		float* d_W, * d_delta;

		CHECK(cudaMalloc((void**)& d_W, n * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_delta, nx * sizeof(float)));
		CHECK(cudaMemcpy(d_W, W, n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_delta, delta, nx * sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(BLOCK_SIZE);
		dim3 grid((n + block.x - 1) / block.x);
		embedCalGradGPU << <grid, block >> > (d_W, d_delta, textCode, nx, ny);
		//sync
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy(out, d_W, n * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(d_W);
		cudaFree(d_delta);
		return out;

	}


	void BasicLayer::randInit(int m ,int n) {

		cudaMalloc((void**)& rndstates, m * n * sizeof(curandState));
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		randInitGPU << <grid,block >> > (n, m, rndstates);
	}

	void BasicLayer::weightbiasTruncInit(float* Wh, float* Wx, float* b,
		const int Whlen, const int Wxlen, const int blen) {

		int m = hiddenStates + embedSize + 1;
		int n = blen;

		randInit(m, n);

		float* d_Wh, * d_Wx, * d_b;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_Wh, Whlen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_Wx, Wxlen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_b, blen * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_Wh, Wh, Whlen * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Wx, Wx, Wxlen * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_b, b, blen * sizeof(float), cudaMemcpyHostToDevice));

		//randomlize weights and bias

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		weightbiasTruncInitGPU << <grid, block >> > (hiddenStates, embedSize,
			d_Wh, d_Wx, d_b, rndstates);

		//transfer data from device to host
		CHECK(cudaMemcpy(Wh, d_Wh, Whlen * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(Wx, d_Wx, Wxlen * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(b, d_b, blen * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_Wh));
		CHECK(cudaFree(d_Wx));
		CHECK(cudaFree(d_b));

	}

	void BasicLayer::weightbiasGradInit(float* Wh, float* Wx, float* b,
		const int Whlen, const int Wxlen, const int blen) {

		float* d_Wh, * d_Wx, * d_b;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_Wh, Whlen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_Wx, Wxlen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_b, blen * sizeof(float)));

		//Init as 0;
		int m = hiddenStates + embedSize + 1;
		int n = blen;

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		weightbiasGradInitGPU << <grid, block >> > (hiddenStates, embedSize,
			d_Wh, d_Wx, d_b);

		//transfer data from device to host
		CHECK(cudaMemcpy(Wh, d_Wh, Whlen * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(Wx, d_Wx, Wxlen * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(b, d_b, blen * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_Wh));
		CHECK(cudaFree(d_Wx));
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

	void BasicLayer::denseweightbiasTruncInit(float* W, float* b,
		const int Wlen, const int blen) {
		int m = categories;
		int n = hiddenStates + 1;

		randInit(m, n);

		float* d_W, * d_b;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_W, Wlen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_b, blen * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_W, W, Wlen * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_b, b, blen * sizeof(float), cudaMemcpyHostToDevice));

		//randomlize weights and bias


		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		denseweightbiasTruncInitGPU << <grid, block >> > (n, m,
			 d_W, d_b, rndstates, 1);

		//transfer data from device to host
		CHECK(cudaMemcpy(W, d_W, Wlen * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(b, d_b, blen * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_W));
		CHECK(cudaFree(d_b));

	}

	void BasicLayer::denseweightbiasGradInit(float* W, float* b,
		const int Wlen, const int blen) {

		int m = categories;
		int n = hiddenStates + 1;

		randInit(m, n);

		float* d_W, * d_b;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_W, Wlen * sizeof(float)));
		CHECK(cudaMalloc((void**)& d_b, blen * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_W, W, Wlen * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_b, b, blen * sizeof(float), cudaMemcpyHostToDevice));

		//randomlize weights and bias


		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		denseweightbiasGradInitGPU << <grid, block >> > (n, m,
			d_W, d_b, rndstates, 1);

		//transfer data from device to host
		CHECK(cudaMemcpy(W, d_W, Wlen * sizeof(float), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(b, d_b, blen * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_W));
		CHECK(cudaFree(d_b));
	}

	void BasicLayer::embedweightTruncInit(float* W, const int Wlen) {

		int m = embedSize;
		int n = Wlen / m;

		randInit(m, n);

		float* d_W;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_W, Wlen * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_W, W, Wlen * sizeof(float), cudaMemcpyHostToDevice));

		//randomlize weights and bias

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);

		embedweightbiasTruncInitGPU << <grid, block >> > (n, m,
			d_W, rndstates);

		//transfer data from device to host
		CHECK(cudaMemcpy(W, d_W, Wlen * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_W));

	}

	void BasicLayer::embedweightGradInit(float* W, const int Wlen) {

		int m = embedSize;
		int n = Wlen / m;

		float* d_W;

		//malloc device memory
		CHECK(cudaMalloc((void**)& d_W, Wlen * sizeof(float)));

		//transfer data from host to device
		CHECK(cudaMemcpy(d_W, W, Wlen * sizeof(float), cudaMemcpyHostToDevice));

		//randomlize weights and bias

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid((n + block.x - 1) / block.x, m);
		denseweightbiasGradInitGPU << <grid, block >> > (n, m,
			d_W, d_W, rndstates, 0);

		//transfer data from device to host
		CHECK(cudaMemcpy(W, d_W, Wlen * sizeof(float), cudaMemcpyDeviceToHost));

		//free device memory
		CHECK(cudaFree(d_W));
	}

}