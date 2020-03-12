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
	void softmax(float* A, int len);
	void tanh(float* A, int len);
	void sigmoid(float* A, int len);
	void tanhPrime(float* matA, int n);
	void sigmoidPrime(float* matA, int n);
	void crossEntropyLoss();
}

#endif /* UTIL_H_ */
