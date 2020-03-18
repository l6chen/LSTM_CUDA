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

#include <algorithm>
#include <sstream>
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

	bool isAlphabetic(char aChar) {
		if ((aChar >= 'a') && (aChar <= 'z'))
			return true;
		else if ((aChar >= 'A') && (aChar <= 'Z'))
			return true;
		else
			return false;
	}

	bool isNumeric(char aChar) {
		if ((aChar >= '0') && (aChar <= '9'))
			return true;
		else
			return false;
	}

	bool isAlphaNumeric(char aChar) {
		if (isAlphabetic(aChar))
			return true;
		else if (isNumeric(aChar))
			return true;
		else
			return false;
	}

	void wash(std::vector<std::string>& texts) {
		std::string starter = "rt";
		std::string ender = "http";
		
		for (auto& text : texts) {
			std::string s = text;
			std::transform(text.begin(), text.end(), text.begin(), ::tolower);
			size_t pos1 = text.find(starter);
			if (pos1 != std::string::npos && pos1 < 4) {
				text = text.substr(pos1 + 2);
			}
			size_t pos2 = text.rfind(ender);
			size_t pos3 = text.find(" ");
			if (pos2 != std::string::npos && pos2 > 4) {
				text = text.substr(0, pos2);
			}
			else if (pos2 == 0 && pos3 != std::string::npos) {
				text = text.substr(pos3);
			}
		}

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

	const std::unordered_map<std::string, int> makeDict(
		const std::vector<std::vector<std::string>> tokens) {
		std::unordered_map<std::string, int> dict;
		int dictSize = 0;
		for (auto sentence : tokens) {
			for (auto word: sentence){
				if (dict.find(word) == dict.end()) {
					dict.insert({ word, dictSize });
					dictSize++;
				}
			}
		}
		return dict;
	}

	const std::vector<std::vector<std::string>> 
	tokenizer(std::vector<std::string> texts) {
		std::vector<std::vector<std::string>> tokens;
		for (auto text : texts) {
			std::string word;
			std::stringstream check(text);
			std::vector<std::string> sentence;
			while (std::getline(check, word, ' ')) {
				if (word.length() > 0) {
					std::string clean;
					for (int i = 0; i < word.length(); i++) {
						if (isAlphaNumeric(word[i])) {
							clean += word[i];
						}
					}
					if (clean.size() > 0) sentence.push_back(clean);
				}
			}
			if (sentence.size() > 0) tokens.push_back(sentence); 
		}
		return tokens;
	}

	const std::vector<std::vector<int>> textEncode(
		const std::vector<std::vector<std::string>> tokens,
		const std::unordered_map<std::string, int> dict) {
		std::vector<std::vector<int>> textCode;
		for (auto token : tokens) {
			std::vector<int> senCode(token.size(), 0);
			for (int i = 0; i < token.size(); i++) {
				std::string t = token[i];
				senCode[i] = dict.at(t);
			}
			textCode.push_back(senCode);
		}
		return textCode;
	}

	const std::vector<int> labelEncode(std::vector<std::string> labels) {
		std::vector<int> labelCode(labels.size());
		for (int i = 0; i < labels.size(); i++) {
			auto label = labels[i];
			if (label == "Positive") {
				labelCode[i] = 0;
			}
			else if (label == "Neutral") {
				labelCode[i] = 1;
			}
			else if (label == "Negative") {
				labelCode[i] = 2;
			}
			else {
				std::cout << "Invalid Label";
				break;
			}
		}

		return labelCode;
	}

	DataSets DataLoader::datasplitter(const std::vector<std::vector<int>> textCode,
		const std::vector<int> labelCode) {

		DataSets splitted;
		if (textCode.size() != labelCode.size()) {
			std::cout << textCode.size() << " " << labelCode.size();
			throw "text and label mismatch!";
		}
		else {
			int sampleSize = labelCode.size();
			int trainSize = (int)sampleSize * _trainPer;
			int testSize = (int)sampleSize * (1 - _trainPer) / 2;
			int valSize = sampleSize - trainSize - testSize;
			
			std::vector<int> draw(sampleSize);
			for (int i = 0; i < sampleSize; i++) {
				draw[i] = i;
			}
			std::random_shuffle(draw.begin(), draw.end());
			for (int i = 0; i < trainSize; i++) {
				splitted.trainX.push_back(textCode[draw[i]]);
				splitted.trainY.push_back(labelCode[draw[i]]);
			}
			for (int i = 0; i < testSize; i++) {
				int drawid = i + trainSize;
				splitted.testX.push_back(textCode[draw[drawid]]);
				splitted.testY.push_back(labelCode[draw[drawid]]);
			}
			for (int i = 0; i < valSize; i++) {
				int drawid = i + trainSize + testSize;
				splitted.valX.push_back(textCode[draw[drawid]]);
				splitted.valY.push_back(labelCode[draw[drawid]]);
			}
		}
		return splitted;
	}

	DataSets DataLoader::load() {
		std::string sentiPath = _datasetDir + "/sentiment.txt";
		std::string textPath = _datasetDir + "/text.txt";
		loadTxt(sentiPath, _sentiments, numOfSamples);
		loadTxt(textPath, _texts, numOfSamples);
		wash(_texts);
		const std::vector<std::vector<std::string>> tokens = tokenizer(_texts);
		const std::unordered_map<std::string, int> dict = makeDict(tokens);
		const std::vector<std::vector<int>> textCode = textEncode(tokens, dict);
		const std::vector<int> labelCode = labelEncode(_sentiments);
		DataSets ds = datasplitter(textCode, labelCode);
		std::cout << ds.trainX[0][0] << " " << ds.trainY[0] << std::endl;//for debug
		std::cout << dict.size() <<std::endl;//for debug
		return ds;
	}

	DataLoader::DataLoader(float trainPer) {
		char buf1[256];
		_getcwd(buf1, sizeof(buf1));
		std::string testDir = buf1;
		testDir = subreplace(testDir, "\\", "/");
		std::string datasetDir = testDir + "/../../../datasets";
		_datasetDir = datasetDir;
		_trainPer = trainPer;
	}

	void DataLoader::writeWashed(std::vector<std::string> washed) {
		std::ofstream os;
		os.open(this->_datasetDir + "/texts_washed.txt");
		for (auto text : washed) {
			os << text << std::endl;
		}
		os.close();
	}

	const std::vector<std::vector<int>> DataLoader::oneHotCoding(std::vector<std::string> labels) {
		std::vector<std::vector<int>> oneHot;
		for (auto label : labels) {
			if (label == "Positive") {
				oneHot.push_back({ 1,0,0 });
			}
			else if (label == "Neutral") {
				oneHot.push_back({ 0,1,0 });
			}
			else if (label == "Negative") {
				oneHot.push_back({ 0,0,1 });
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