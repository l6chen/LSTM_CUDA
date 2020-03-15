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

	//All matrix is m by n 
	
	float* matElem(const float* matA, const float* matB, int m, int n, char op);
	float* matTrans(const float* matA, int height, int width);
	float* matMul(const float* matA, const float* matB, int m, int n, int k);
	float* matMulScal(const float* matA, float scal, int m, int n);
	void matMul_inplace(float* out, float* matA, float* matB, int m, int n, int k);//[m,n] * [n,k]
	void matElem_inplace(float* matA, float* matB, int m, int n, char op);
	void matTrans_inplace(float* matA, int height, int width);
	void softmax(float* A, int n, const int categories);
	void tanh(float* A, int n);
	void sigmoid(float* A, int n);
	void tanhPrime(float* matA, int n);
	void sigmoidPrime(float* matA, int n);
	float crossEntropyLoss(float* pred, float* oneHot, int m, int n, bool iftest = false);
}

#endif /* UTIL_H_ */
