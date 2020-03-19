/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <iostream>
#include <string>
#include "lstm.h"
#include "dataLoader.h"

int main() {
	std::cout << "LSTM STRATS." << std::endl;
	const std::string DELIMITER = "================================================================================";
	dataLoader::DataLoader loader(0.8);
	dataLoader::DataSets* ds = loader.load();
	std::cout << "Data Loaded" << "\n" << DELIMITER <<"\n";
	lstm::LSTMNetwork* model = new lstm::LSTMNetwork(0.001, 3, 128, 10, 50);
	model->train(ds);
	delete model;
	system("PAUSE");
	return 0;
}
