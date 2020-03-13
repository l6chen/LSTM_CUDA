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
#include <iomanip>
#include <direct.h>

std::string subreplace(std::string resource_str, std::string sub_str, std::string new_str)
{
	std::string::size_type pos = 0;
	while ((pos = resource_str.find(sub_str)) != std::string::npos) 
	{
		resource_str.replace(pos, sub_str.length(), new_str);
	}
	return resource_str;
}


int main() {
	const std::string DELIMITER = "================================================================================";
	std::cout << "Test begins." << std::endl;

	//Test Dataloader
	std::cout << DELIMITER << std::endl << "Test Data Loader" << std::endl;
	char buf1[256];
	_getcwd(buf1, sizeof(buf1));
	std::string testDir = buf1;
	testDir = subreplace(testDir, "\\", "/");
	std::string datasetDir = testDir + "/../../../datasets";
	dataLoader::DataLoader loader(datasetDir);
	loader.load();
	std::vector<std::string> sentiments = loader.getsentiments();
	std::vector<std::string> tweets = loader.gettexts();
	for (int i = 0; i < 5; i++)
		std::cout << sentiments[i] << " " << tweets[i] << std::endl;
	std::cout << *(sentiments.end() - 1) << " " << *(tweets.end() - 1) << std::endl;

	//Test Elementwise operations
	std::cout << DELIMITER << std::endl << "Test Elementwise operations" << std::endl;
	testUtil::testmatrixCalElem('+');
	testUtil::testmatrixCalElem('-');
	testUtil::testmatrixCalElem('*');
	//testUtil::testmatrixCalElem('/');//not supported

	//Test Matrix Mul
	std::cout << DELIMITER << std::endl << "Test Matrix Mul" << std::endl;
	testUtil::testmatrixMul();
	

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
	testLayer::testLayerInit();


	std::cout << "Test ends.";
	system("PAUSE");
	return 0;
}