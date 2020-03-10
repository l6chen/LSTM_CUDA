/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <iostream>
#include <string>
#include <vector>
#include "../lstm/dataLoader.h"

int main() {
	const std::string DELIMITER = "================================================================================";
	std::cout << "Test begins." << std::endl;

	//Test Dataloader
	std::cout << DELIMITER << std::endl << "Test Data Loader" << std::endl;
	std::string datasetDir = "C:/Users/Morligan/Desktop/Githubprj/CUDA/LSTM_CUDA/datasets";
	dataLoader::DataLoader loader(datasetDir);
	loader.load();
	std::vector<std::string> sentiments = loader.getsentiments();
	std::vector<std::string> tweets = loader.gettexts();
	for (int i = 0; i < 5; i++)
		std::cout << sentiments[i] << " " << tweets[i] << std::endl;
	std::cout << *(sentiments.end() - 1) << " " << *(tweets.end() - 1) << std::endl;
	std::cout << DELIMITER << std::endl;
	std::cout << "Test ends.";
	system("PAUSE");
	return 0;
}