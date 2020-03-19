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
#include "test_Layer.h"




int main() {
	const std::string DELIMITER = "================================================================================";
	std::cout << "Test begins." << std::endl;

	//Test Dataloader
	std::cout << DELIMITER << std::endl << "Test Data Loader" << std::endl;

	dataLoader::DataLoader loader(0.8);
	dataLoader::DataSets* ds = loader.load();

	//Test Elementwise operations
	std::cout << DELIMITER << std::endl << "Test Elementwise operations" << std::endl;
	testUtil::testmatElem('+');
	testUtil::testmatElem('-');
	testUtil::testmatElem('*');
	//testUtil::testmatElem('/');//not supported

	//Test Matrix Mul
	std::cout << DELIMITER << std::endl << "Test Matrix Mul" << std::endl;
	testUtil::testmatMul();

	//Test Matrix Scal Mul
	std::cout << DELIMITER << std::endl << "Test Matrix Scal Mul" << std::endl;
	testUtil::testmatMulScal();
	
	//Test transpose
	std::cout << DELIMITER << std::endl << "Test Matrix Tanspose" << std::endl;
	testUtil::testmatTranspose();	

	//Test tanh
	std::cout << DELIMITER << std::endl << "Test tanh" << std::endl;
	testUtil::testtanh();

	//Test softmax
	std::cout << DELIMITER << std::endl << "Test softmax" << std::endl;
	testUtil::testsoftmax();

	//test sigmoid
	std::cout << DELIMITER << std::endl << "Test sigmoid" << std::endl;
	testUtil::testsigmoid();

	//Test Tanh Prime
	std::cout << DELIMITER << std::endl << "Test Tanh Prime" << std::endl;
	testUtil::testtanhPrime();

	//Test Sigmoid Prime
	std::cout << DELIMITER << std::endl << "Test Sigmoid Prime" << std::endl;
	testUtil::testsigmoidPrime();

	//Test Cross Entropy Loss
	std::cout << DELIMITER << std::endl << "Test Cross Entropy Loss" << std::endl;
	testUtil::testcrossEntropyLoss();

	//Test Basic Layer
	std::cout << DELIMITER << std::endl << "Test Basic Layer" << std::endl;
	testLayer::testBasicLayer();

	//Test Gate Layer
	std::cout << DELIMITER << std::endl << "Test Gate Layer" << std::endl;
	testLayer::testGateLayer();

	//Test Dense Layer
	std::cout << DELIMITER << std::endl << "Test Dense Layer" << std::endl;
	testLayer::testDenseLayer();
	
	//Test Embed Layer
	std::cout << DELIMITER << std::endl << "Test Embed Layer" << std::endl;
	testLayer::testEmbedLayer();

	std::cout << "Test ends.";
	system("PAUSE");
	return 0;
}