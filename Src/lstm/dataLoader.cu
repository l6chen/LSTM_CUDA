/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "dataLoader.h"
#define numOfSamples 13871

namespace dataLoader {

	void loadTxt(std::string filePath, std::vector<std::string>& txt, int samples) {
		std::fstream fs(filePath.c_str(), std::fstream::in);
		int count = 0;
		if (fs.good()) {
			std::string buf;
			while (std::getline(fs, buf) && count < samples) {
				txt.push_back(buf);
				++count;
			}
		}
		fs.close();
	}

	void DataLoader::load() {
		std::string sentiPath = this->_datasetDir + "/sentiment.txt";
		std::string textPath = this->_datasetDir + "/text.txt";
		loadTxt(sentiPath, this->_sentiments, numOfSamples);
		loadTxt(textPath, this->_texts, numOfSamples);
	}
}