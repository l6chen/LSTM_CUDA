/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include "test_util.h"
#include <string>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define CATEGORIES 3
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))


namespace testUtil {
	void initialData(float* ip, const int size)
	{

		for (int i = 0; i < size; i++)
		{
			ip[i] = (float)(rand() & 0xFF % 100) / 100.0f;
		}

		return;
	}

	void checkResult(float* hostRef, float* gpuRef, const int N, std::string testtype)
	{
		double epsilon = 1.0E-6;
		bool match = 1;

		for (int i = 0; i < N; i++)
		{
			if (abs(hostRef[i] - gpuRef[i]) > epsilon)
			{
				match = 0;
				printf("idx: %d host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
				break;
			}
		}

		if (match)
			std::cout << "Arrays match for " << testtype << "\n\n";
		else
			std::cout << "Arrays do not match for " << testtype << "\n\n";
	}


	void transposeHost(const float *A, float *B, const int nrows, const int ncols)
	{
		for (int iy = 0; iy < nrows; ++iy)
		{
			for (int ix = 0; ix < ncols; ++ix)
			{
				B[INDEX(ix, iy, nrows)] = A[INDEX(iy, ix, ncols)];
			}
		}
	}








	void matrixCalElemOnHost(const float* A, const float* B, float* C, const int nx,
		const int ny, char op)
	{

		float* ic = C;

		for (int iy = 0; iy < ny; iy++)
		{
			for (int ix = 0; ix < nx; ix++)
			{
				switch (op)
				{
				case '+':
					ic[ix] = A[ix] + B[ix];
					break;
				case '-':
					ic[ix] = A[ix] - B[ix];
					break;
				case '*':
					ic[ix] = A[ix] * B[ix];
					break;
				}
			}

			A += nx;
			B += nx;
			ic += nx;
		}

		return;
	}

	void mulMatrixOnHost(const float* A, const float* B, float* C, const int nx,
		const int ny, const int nz)
	{
		float* ic = C;

		for (int iy = 0; iy < ny; iy++)
		{
			for (int iz = 0; iz < nz; iz++)
			{
				float sum = 0;
				for (int ix = 0; ix < nx; ix++)
					sum += A[iy * nx + ix] * B[ix * nz + iz];
				ic[iy * nz + iz] = sum;
			}
		}

		return;
	}


	void tanhOnHost(const float *A, float*B, int nx) {

		float* ib = B;

		for (int ix = 0; ix < nx; ix++)
		{
			
			ib[ix] =  (expf(A[ix]) - expf(-A[ix])) / (expf(A[ix]) + expf(-A[ix]));
		}

		return;
	}

	void softmaxOnHost(const float *A, float*B, int nx) {
		
		float* ib = B;
		float* ic;
		ic = (float*)malloc((nx / 3) * sizeof(float));

		// compute sum
		float sum = 0.0;
		for (int ix = 0; ix < nx; ix++) {
			ib[ix] = expf(A[ix]);
		}

		for (int ix = 0; ix < nx; ix++) {
			sum += ib[ix];
			if((ix + 1) % CATEGORIES == 0) {
				ic[ix / 3] = sum;
				sum = 0.0;
			}
		}

		for (int ix = 0; ix < nx; ix++) {

			ib[ix] = ib[ix] / ic[ix / 3];
		}

		free(ic);
		ic = nullptr;
	}
	
	
	
	void sigmoidOnHost(const float *A, float*B, int nx) {
		
		float* ib = B;

		for (int ix = 0; ix < nx; ix++)
		{

			ib[ix] = expf(A[ix]) / (1.0f + expf(A[ix]));
		}

		return;
	}

	void tanhPrimeOnHost(const float* A, float* B, const int nx)
	{
		float* ib = B;

		for (int ix = 0; ix < nx; ix++)
			ib[ix] = 1.0f - A[ix] * A[ix];

		return;
	}

	void sigmoidPrimeOnHost(const float* A, float* B, const int nx)
	{
		float* ib = B;

		for (int ix = 0; ix < nx; ix++)
			ib[ix] = A[ix] * (1.0f - A[ix]);

		return;
	}

	float crossEntropyLossOnHost(float* matA, float* matB, const int nx, const int ny) {
		float* ia = matA;
		float* ib = matB;
		
		float* ic;
		ic = (float*)malloc(ny * sizeof(float));

		for (int iy = 0; iy < ny; iy++) {
			ic[iy] = 0;
			for (int ix = 0; ix < nx; ix++) {
				int curIdx = iy * nx + ix;
				if (ib[curIdx] != 0 && ia[curIdx] != 0) {
					ic[iy] += -logf(ia[curIdx]);
				}
			}

		}

		float avgcost = 0;
		for (int i = 0; i < ny; i++) {
			avgcost += ic[i];
			//std::cout << ic[iy];
		}
		avgcost /= ny;
		free(ic);
		ic = nullptr;

		return avgcost;
	}


	void testmatrixCalElem(char op) {
		std::string opStr = std::string(1, op);
		std::string testtype = "Elem " + opStr;
		int nx = 1 << 5;
		int ny = 1 << 5;

		int nxy = nx * ny;
		int nBytes = nxy * sizeof(float);

		float* matA, * matB, * cpuM;
		matA = (float*)malloc(nBytes);
		matB = (float*)malloc(nBytes);
		cpuM = (float*)malloc(nBytes);

		initialData(matA, nxy);
		initialData(matB, nxy);

		const float* tmpA = matA;
		const float* tmpB = matB;

		matrixCalElemOnHost(tmpA, tmpB, cpuM, nx, ny, op);
		util::matrixCalElem(matA, matB, ny, nx, op);
		checkResult(cpuM, matA, nxy, testtype);

		// free host memory
		free(matA);
		free(matB);
		free(cpuM);
		tmpA = nullptr;
		tmpB = nullptr;

		// reset device
		CHECK(cudaDeviceReset());

	}

	void testmatrixMul() {

		std::string testtype = "Mul";
		int ny = 1 << 3;
		int nx = 1 << 4;
		int nz = 1;

		int nxy = nx * ny, nyz = ny * nz, nxz = nx * nz;
		int nxyB = nxy * sizeof(float), nyzB = nyz * sizeof(float),
			nxzB = nxz * sizeof(float);

		float* matA, * matB, * cpuM, *gpuM;
		matA = (float*)malloc(nxyB);
		matB = (float*)malloc(nxzB);
		cpuM = (float*)malloc(nyzB);
		gpuM = (float*)malloc(nyzB);

		initialData(matA, nxy);
		initialData(matB, nxz);

		const float* tmpA = matA;
		const float* tmpB = matB;

		mulMatrixOnHost(tmpA, tmpB, cpuM, nx, ny, nz);
		util::matrixMul(gpuM, matA, matB, ny, nx, nz);
		checkResult(cpuM, gpuM, nyz, testtype);

		// free host memory
		free(matA);
		free(matB);
		free(cpuM);
		free(gpuM);
		tmpA = nullptr;
		tmpB = nullptr;

		// reset device
		CHECK(cudaDeviceReset());

	}

	void testtanh() {

		std::string testtype = "tanh";
		int nx = 1 << 5;
		int nBytes = nx * sizeof(float);

		float* matA, *cpuM;
		matA = (float*)malloc(nBytes);
		cpuM = (float*)malloc(nBytes);

		initialData(matA, nx);

		const float* tmpA = matA;

		tanhOnHost(tmpA, cpuM, nx);
		util::tanh(matA, nx);
		checkResult(cpuM, matA, nx, testtype);


		// free host memory
		free(matA);
		free(cpuM);
		tmpA = nullptr;

		// reset device
		CHECK(cudaDeviceReset());
	}

	void testsoftmax() {

		std::string testtype = "softmax";
		const int categories = 3;
		int nx = 3 * (1 << 5);
		int nBytes = nx * sizeof(float);

		float* matA, * cpuM;
		matA = (float*)malloc(nBytes);
		cpuM = (float*)malloc(nBytes);

		initialData(matA, nx);

		const float* tmpA = matA;

		softmaxOnHost(tmpA, cpuM, nx);
		util::softmax(matA, nx, categories);
		checkResult(cpuM, matA, nx, testtype);

		// free host memory
		free(matA);
		free(cpuM);
		tmpA = nullptr;

		// reset device
		CHECK(cudaDeviceReset());
	}

	void testsigmoid() {

		std::string testtype = "sigmoid";
		int nx = 1 << 5;
		int nBytes = nx * sizeof(float);


		float* matA, * cpuM;
		matA = (float*)malloc(nBytes);
		cpuM = (float*)malloc(nBytes);

		initialData(matA, nx);

		const float* tmpA = matA;

		sigmoidOnHost(tmpA, cpuM, nx);
		util::sigmoid(matA, nx);
		checkResult(cpuM, matA, nx, testtype);

		// free host memory
		free(matA);
		free(cpuM);
		tmpA = nullptr;

		// reset device
		CHECK(cudaDeviceReset());
	}


	void testtanhPrime() {
		std::string testtype = "tanhprime";
		int nx = 1 << 5;

		int nBytes = nx * sizeof(float);

		float* matA, * cpuM;
		matA = (float*)malloc(nBytes);
		cpuM = (float*)malloc(nBytes);

		initialData(matA, nx);

		const float* tmpA = matA;

		tanhPrimeOnHost(tmpA, cpuM, nx);
		util::tanhPrime(matA, nx);
		checkResult(cpuM, matA, nx, testtype);

		// free host memory
		free(matA);
		free(cpuM);
		tmpA = nullptr;

		// reset device
		CHECK(cudaDeviceReset());

	}

	void testsigmoidPrime() {
		std::string testtype = "sigmoidprime";
		int nx = 1 << 5;

		int nBytes = nx * sizeof(float);

		float* matA, * cpuM;
		matA = (float*)malloc(nBytes);
		cpuM = (float*)malloc(nBytes);

		initialData(matA, nx);

		const float* tmpA = matA;

		sigmoidPrimeOnHost(tmpA, cpuM, nx);
		util::sigmoidPrime(matA, nx);
		checkResult(cpuM, matA, nx, testtype);

		// free host memory
		free(matA);
		free(cpuM);
		tmpA = nullptr;

		// reset device
		CHECK(cudaDeviceReset());
	}

	void testcrossEntropyLoss() {
		std::string testtype = "crossEntropyLoss";
		int nx = 1 << 5;
		int ny = 1 << 5;

		int nxy = nx * ny;
		int nBytes = nxy * sizeof(float);

		float* matA, * matB;
		matA = (float*)malloc(nBytes);
		matB = (float*)malloc(nBytes);

		initialData(matA, nxy);
		initialData(matB, nxy);

		float cpuLoss = crossEntropyLossOnHost(matA, matB, nx, ny);
		float gpuLoss = util::crossEntropyLoss(matA, matB, ny, nx, true);

		if (abs(cpuLoss - gpuLoss) > 1.0E-3) {
			std::cout << testtype << " fails" << std::endl;
			std::cout << "cpuLoss:" << cpuLoss << " gpuLoss:" << gpuLoss << std::endl;
			std::cout << "diff: " << cpuLoss - gpuLoss << "\n\n";
		}
		else {
			std::cout << testtype << " success" << "\n\n";
		}
			

		// free host memory
		free(matA);
		free(matB);

		// reset device
		CHECK(cudaDeviceReset());
	}


	void testmatTranspose() {

		std::string testtype = "Transpose";
		int ny = 1 << 3;
		int nx = 1 << 4;

		int nxy = nx * ny;
		int nBytes = nxy * sizeof(float);

		float* matA, *cpuM;
		matA = (float*)malloc(nBytes);
		cpuM = (float*)malloc(nBytes);


		initialData(matA, nxy);

		const float* tmpA = matA;

		transposeHost(tmpA, cpuM, ny, nx);
		util::matrixTranspose(matA, ny, nx);
		checkResult(cpuM, matA, nxy, testtype);


		// free host memory
		free(matA);
		tmpA = nullptr;

		// reset device
		CHECK(cudaDeviceReset());

	}
}
