/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <string>
#include <vector>

namespace dataLoader {
	class DataLoader {
	private:
		std::string _datasetDir;
		std::vector<std::string> _sentiments;
		std::vector<std::string> _texts;
	public:
		DataLoader(std::string datasetDir) { _datasetDir = datasetDir; }
		void load();
		std::vector<std::string> getsentiments() { return _sentiments; }
		std::vector<std::string> gettexts() { return _texts; }
	};
}

#endif /* DATALOADER_H_ */

