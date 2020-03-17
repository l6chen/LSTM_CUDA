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

	std::string subreplace(std::string resource_str, std::string sub_str, std::string new_str)
	{
		std::string::size_type pos = 0;
		while ((pos = resource_str.find(sub_str)) != std::string::npos)
		{
			resource_str.replace(pos, sub_str.length(), new_str);
		}
		return resource_str;
	}

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

	DataLoader::DataLoader() {
		char buf1[256];
		_getcwd(buf1, sizeof(buf1));
		std::string testDir = buf1;
		testDir = subreplace(testDir, "\\", "/");
		std::string datasetDir = testDir + "/../../../datasets";
		_datasetDir = datasetDir;
	}

	const std::vector<std::vector<int>> DataLoader::oneHotCoding(std::vector<std::string> labels) {
		std::vector<std::vector<int>> oneHot;
		for (auto label : labels) {
			if (label == "Positive") {
				oneHot.push_back({1,0,0});
			}
			else if (label == "Neutral") {
				oneHot.push_back({0,1,0});
			}
			else if (label == "Negative") {
				oneHot.push_back({0,0,1});
			}
			else {
				std::cout << "Invalid Label";
				break;
			}
		}

		const std::vector<std::vector<int>> constoneHot = oneHot;
		return constoneHot;
	}
}