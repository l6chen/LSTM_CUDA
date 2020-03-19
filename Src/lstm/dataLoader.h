/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <string>
#include <vector>
#include <iomanip>
#include <direct.h>
#include <unordered_map>

namespace dataLoader {
	class DataSets {
	public:
		std::vector<std::vector<int>> trainX, testX, valX;
		std::vector<int> trainY, testY, valY;
		int dictLen;
		int sentenLen;
	};
	class DataLoader {
	private:
		std::string _datasetDir;
		std::vector<std::string> _sentiments;
		std::vector<std::string> _texts;
		float _trainPer;
		const std::vector<std::vector<int>> oneHotCoding(std::vector<std::string> labels);//deprecated
		void writeWashed(std::vector<std::string> washed);
		DataSets* datasplitter(const std::vector<std::vector<int>> textCode,
			const std::vector<int> labelCode);
	public:
		DataLoader(float trainPer = 0.8);
		~DataLoader(){}
		std::vector<std::string> getsentiments() { return _sentiments; }
		std::vector<std::string> gettexts() { return _texts; }
		DataSets* load();
	};
}

#endif /* DATALOADER_H_ */

