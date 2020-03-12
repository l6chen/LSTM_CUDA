/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

#include "../lstm/util.h"

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

namespace testUtil {
	void testmatrixSum();
	void testmatrixMulElem();
	void testmatrixMul();
	void testtanh();
	void testsigmoid();
	void testsoftmax();
	void testmatrixMul();
	void testtanhPrime();
	void testsigmoidPrime();
	void testcrossEntropyLoss();
}

#endif /* TEST_UTIL_H_ */