/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <iostream>
#include <string>
#include "lstm.h"
#include "dataLoader.h"
#include <fstream>

int main() {
	const std::string DELIMITER = "================================================================================";
	std::ofstream out("C:/Users/Morligan/Desktop/Githubprj/CUDA/LSTM_CUDA/datasets/out.txt");
	std::streambuf* coutbuf = std::cout.rdbuf(); //save old buf
	//std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
	std::cout << "LSTM STRATS." << std::endl;
	dataLoader::DataLoader* loader = new dataLoader::DataLoader(0.8);
	dataLoader::DataSets* ds = loader->load();
	std::cout << "Data Loaded" << "\n" << DELIMITER <<"\n";
	lstm::LSTMNetwork* model = new lstm::LSTMNetwork(0.001, 3, 128, 10, 50);
	model->train(ds);
	std::cout << "Train Finished" << "\n" << DELIMITER << "\n";
	delete model;
	system("PAUSE");
	return 0;
}
