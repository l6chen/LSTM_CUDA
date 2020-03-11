/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "test_util.h"
#include "../common/common.h"


/*
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

	void checkResult(float* hostRef, float* gpuRef, const int N)
	{
		double epsilon = 1.0E-8;
		bool match = 1;

		for (int i = 0; i < N; i++)
		{
			if (abs(hostRef[i] - gpuRef[i]) > epsilon)
			{
				match = 0;
				printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
				break;
			}
		}

		if (match)
			printf("Arrays match.\n\n");
		else
			printf("Arrays do not match.\n\n");
	}

	void testmatrixSum() {

		int nx = 1 << 14;
		int ny = 1 << 14;

		int nxy = nx * ny;
		int nBytes = nxy * sizeof(float);
		/*
		float* matA, * matB, * matC;
		matA = (float*)malloc(nBytes);
		matB = (float*)malloc(nBytes);
		matC = (float*)malloc(nBytes);

		initialData(matA, nxy);
		initialData(matB, nxy);

		sumMatrixOnHost(matA, matB, matC, nx, ny);
		util::matrixSum(matA, matB, ny, nx);
		checkResult(matC, matA, nxy);

		// free host memory
		free(matA);
		free(matB);
		free(matC);

		// reset device
		CHECK(cudaDeviceReset());
		*
	}
}
*/