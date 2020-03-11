/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef UTIL_H_
#define UTIL_H_

#include <vector>
#include <iostream>


namespace util {
	void matrixSum(float* matA, float* matB, int m, int n);//m by n matrix sum
	void matrixMul(float* out, float* matA, float* matB, int m, int n, int k);//[m,n] * [n,k]
	void matrixMulElem(float* matA, float* matB, int m, int n);
	void softmax();
	void tanh();
	void sigmoid();
}

#endif /* UTIL_H_ */
