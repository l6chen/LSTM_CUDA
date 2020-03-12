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

namespace testUtil {
	void initialData(float* ip, const int size)
	{

		for (int i = 0; i < size; i++)
		{
			ip[i] = (float)(rand() & 0xFF) / 10.0f;
		}

		return;
	}

	void sumMatrixOnHost(float* A, float* B, float* C, const int nx,
		const int ny)
	{
		float* ia = A;
		float* ib = B;
		float* ic = C;

		for (int iy = 0; iy < ny; iy++)
		{
			for (int ix = 0; ix < nx; ix++)
			{
				ic[ix] = ia[ix] + ib[ix];

			}

			ia += nx;
			ib += nx;
			ic += nx;
		}

		return;
	}

	void mulElemMatrixOnHost(float* A, float* B, float* C, const int nx,
		const int ny)
	{
		float* ia = A;
		float* ib = B;
		float* ic = C;

		for (int iy = 0; iy < ny; iy++)
		{
			for (int ix = 0; ix < nx; ix++)
			{
				ic[ix] = ia[ix] * ib[ix];

			}

			ia += nx;
			ib += nx;
			ic += nx;
		}

		return;
	}

	void mulMatrixOnHost(float* A, float* B, float* C, const int nx,
		const int ny, const int nz)
	{
		float* ia = A;
		float* ib = B;
		float* ic = C;

		for (int iy = 0; iy < ny; iy++)
		{
			for (int iz = 0; iz < nz; iz++)
			{
				float sum = 0;
				for (int ix = 0; ix < nx; ix++)
					sum += ia[iy * nx + ix] * ib[ix * nz + iz];
				ic[iy * nz + iz] = sum;
			}
		}

		return;
	}


	void tanhOnHost(float *A, float*B, int nx) {
		float* ia = A;
		float* ib = B;

		for (int ix = 0; ix < nx; ix++)
		{
			
			ib[ix] =  (expf(ia[ix]) - expf(-ia[ix])) / (expf(ia[ix]) + expf(-ia[ix]));
		}

		return;
	}

	void softmaxOnHost(float *A, float*B, int nx) {
		
		float* ia = A;
		float* ib = B;
		float* ic;
		ic = (float*)malloc(nx * sizeof(float));

		// compute sum
		float sum = 0.0;
		for (int ix = 0; ix < nx; ix++) {
			ia[ix] = expf(ia[ix]);
		}

		for (int ix = 0; ix < nx; ix++) {
			if((ix+1)%CATEGORIES == 0) {
				ic[ix/3] = sum;
				sum = 0.0;
			}
			sum += ia[ix];
		}

		for (int ix = 0; ix < nx; ix++) {

			ib[ix] = ia[ix] / ic[ix/3];
		}

		free(ic);
	}
	
	
	
	void sigmoidOnHost(float *A, float*B, int nx) {
		
		float* ia = A;
		float* ib = B;

		for (int ix = 0; ix < nx; ix++)
		{

			ib[ix] = expf(ia[ix]) / (1.0f + expf(ia[ix]));
		}

		return;
	}

	void tanhPrimeOnHost(float* A, float* B, const int nx)
	{
		float* ia = A;
		float* ib = B;

		for (int ix = 0; ix < nx; ix++)
			ib[ix] = 1.0f - ia[ix] * ia[ix];

		return;
	}

	void sigmoidPrimeOnHost(float* A, float* B, const int nx)
	{
		float* ia = A;
		float* ib = B;

		for (int ix = 0; ix < nx; ix++)
			ib[ix] = ia[ix] * (1.0f - ia[ix]);

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
				printf("idx: %d host %f gpu %f\n",i, hostRef[i], gpuRef[i]);
				break;
			}
		}

		if (match)
			std::cout << "Arrays match for " << testtype << "\n\n";
		else
			std::cout << "Arrays do not match for " << testtype << "\n\n";
	}

	void testmatrixSum() {

		std::string testtype = "Sum";
		int nx = 1 << 14;
		int ny = 1 << 14;

		int nxy = nx * ny;
		int nBytes = nxy * sizeof(float);
		
		float* matA, * matB, * matC;
		matA = (float*)malloc(nBytes);
		matB = (float*)malloc(nBytes);
		matC = (float*)malloc(nBytes);

		initialData(matA, nxy);
		initialData(matB, nxy);

		sumMatrixOnHost(matA, matB, matC, nx, ny);
		util::matrixSum(matA, matB, ny, nx);
		checkResult(matC, matA, nxy, testtype);

		// free host memory
		free(matA);
		free(matB);
		free(matC);

		// reset device
		CHECK(cudaDeviceReset());
		
	}

	void testmatrixMulElem() {

		std::string testtype = "MulElem";
		int nx = 1 << 14;
		int ny = 1 << 14;

		int nxy = nx * ny;
		int nBytes = nxy * sizeof(float);

		float* matA, * matB, * matC;
		matA = (float*)malloc(nBytes);
		matB = (float*)malloc(nBytes);
		matC = (float*)malloc(nBytes);

		initialData(matA, nxy);
		initialData(matB, nxy);

		mulElemMatrixOnHost(matA, matB, matC, nx, ny);
		util::matrixMulElem(matA, matB, ny, nx);
		checkResult(matC, matA, nxy, testtype);

		// free host memory
		free(matA);
		free(matB);
		free(matC);

		// reset device
		CHECK(cudaDeviceReset());

	}

	void testmatrixMul() {

		std::string testtype = "Mul";
		int ny = 1 << 3;
		int nx = 1 << 4;
		int nz = 1 << 5;

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

		mulMatrixOnHost(matA, matB, cpuM, nx, ny, nz);
		util::matrixMul(gpuM, matA, matB, ny, nx, nz);
		checkResult(cpuM, gpuM, nyz, testtype);

		// free host memory
		free(matA);
		free(matB);
		free(cpuM);
		free(gpuM);

		// reset device
		CHECK(cudaDeviceReset());

	}

	void testtanh() {

		std::string testtype = "tanh";
		int nx = 1 << 14;
		int nBytes = nx * sizeof(float);

		float* A, *B;
		A = (float*)malloc(nBytes);
		B = (float*)malloc(nBytes);

		initialData(A, nx);
		tanhOnHost(A, B, nx);

		util::tanh(A, nx);
		checkResult(B, A, nx, testtype);


		// free host memory
		free(A);
		free(B);

		// reset device
		CHECK(cudaDeviceReset());
	}

	void testsoftmax() {

		std::string testtype = "softmax";
		int nx = 1 << 14;
		int nBytes = nx * sizeof(float);

		float* A, *B;
		A = (float*)malloc(nBytes);
		B = (float*)malloc(nBytes);

		initialData(A, nx);
		softmaxOnHost(A, B, nx);

		util::softmax(A, nx);
		checkResult(B, A, nx, testtype);



		// free host memory
		free(A);
		free(B);

		// reset device
		CHECK(cudaDeviceReset());
	}

	void testsigmoid() {

		std::string testtype = "sigmoid";
		int nx = 1 << 14;
		int nBytes = nx * sizeof(float);

		float* A, *B;
		A = (float*)malloc(nBytes);
		B = (float*)malloc(nBytes);

		initialData(A, nx);
		sigmoidOnHost(A, B, nx);

		util::sigmoid(A, nx);
		checkResult(B, A, nx, testtype);


		// free host memory
		free(A);
		free(B);

		// reset device
		CHECK(cudaDeviceReset());
	}


	void testtanhPrime() {
		std::string testtype = "tanhprime";
		int nx = 1 << 5;

		int nxB = nx * sizeof(float);

		float* matA, *cpuM;
		matA = (float*)malloc(nxB);
		cpuM = (float*)malloc(nxB);

		initialData(matA, nx);
		tanhPrimeOnHost(matA, cpuM, nx);
		util::tanhPrime(matA, nx);
		checkResult(cpuM, matA, nx, testtype);

		free(matA);
		free(cpuM);

		// reset device
		CHECK(cudaDeviceReset());

	}

	void testsigmoidPrime() {
		std::string testtype = "sigmoidprime";
		int nx = 1 << 5;

		int nxB = nx * sizeof(float);

		float* matA, *cpuM;
		matA = (float*)malloc(nxB);
		cpuM = (float*)malloc(nxB);

		initialData(matA, nx);
		sigmoidPrimeOnHost(matA, cpuM, nx);
		util::sigmoidPrime(matA, nx);
		checkResult(cpuM, matA, nx, testtype);

		free(matA);
		free(cpuM);

		// reset device
		CHECK(cudaDeviceReset());
	}



}
