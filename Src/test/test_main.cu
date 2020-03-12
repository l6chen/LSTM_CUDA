/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <iostream>
#include <string>
#include <vector>
#include "../lstm/dataLoader.h"
#include "test_util.h"

int main() {
	const std::string DELIMITER = "================================================================================";
	std::cout << "Test begins." << std::endl;


	//Test Dataloader
	std::cout << DELIMITER << std::endl << "Test Data Loader" << std::endl;
	std::string datasetDir = "C:/Users/p3li/Downloads/LSTM_CUDA-master/LSTM_CUDA-master/datasets";
	dataLoader::DataLoader loader(datasetDir);
	loader.load();
	std::vector<std::string> sentiments = loader.getsentiments();
	std::vector<std::string> tweets = loader.gettexts();
	for (int i = 0; i < 5; i++)
		std::cout << sentiments[i] << " " << tweets[i] << std::endl;
	std::cout << *(sentiments.end() - 1) << " " << *(tweets.end() - 1) << std::endl;
/*
	//Test Matrix Sum
	std::cout << DELIMITER << std::endl << "Test Matrix Sum" << std::endl;
	testUtil::testmatrixSum();

	//Test Matrix Elem Mul
	std::cout << DELIMITER << std::endl << "Test Matrix Elem Mul" << std::endl;
	testUtil::testmatrixMulElem();

	//Test Matrix Mul
	std::cout << DELIMITER << std::endl << "Test Matrix Mul" << std::endl;
	testUtil::testmatrixMul();
	*/

	//Test tanh
	std::cout << DELIMITER << std::endl << "Test tanh" << std::endl;
	testUtil::testtanh();

	//Test softmax
	std::cout << DELIMITER << std::endl << "Test softmax" << std::endl;
	testUtil::testsoftmax();

	//test sigmoid
	std::cout << DELIMITER << std::endl << "Test sigmoid" << std::endl;
	testUtil::testsigmoid();






	std::cout << "Test ends.";
	system("PAUSE");
	return 0;
}