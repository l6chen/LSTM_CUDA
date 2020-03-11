/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <vector>
#include <iostream>

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include "util.h"

namespace util {
	__global__ void cudaHello()
	{
		printf("Hello World from GPU!\n");
	}

	void hellofromGPU()
	{
		cudaHello << <1, 10 >> > ();
	}
}