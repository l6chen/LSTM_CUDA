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

namespace dataLoader {
	class DataLoader {
	private:
		std::string _datasetDir;
		std::vector<std::string> _sentiments;
		std::vector<std::string> _texts;
	public:
		DataLoader();
		DataLoader(std::string datasetDir) : _datasetDir(datasetDir) {}
		void load();
		std::vector<std::string> getsentiments() { return _sentiments; }
		std::vector<std::string> gettexts() { return _texts; }
		const std::vector<std::vector<int>> oneHotCoding(std::vector<std::string> labels);
	};
}

#endif /* DATALOADER_H_ */

