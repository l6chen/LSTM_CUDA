/**
*	Author: Lingfeng Chen, Pengkun Li
*   PID: A53270085, A53270024
*	This file defines the data structure for grid
*/
#include "lstm.h"
#include <unordered_map>

namespace lstm {
	LSTMNetwork::LSTMNetwork(float lr, int cat, int hid, int emb, int ep) {
		lrate = lr;
		categories = cat;
		hiddenStates = hid;
		embedSize = emb;
		epoch = ep;
	}
	void LSTMNetwork::train(dataLoader::DataSets* ds){
		timeSteps = ds->sentenLen;
		OutputsDeltaInit();
		dictSize = ds->dictLen;

		//Layers Init
		embedLayer::EmbedLayer* layerEmbed = new embedLayer::EmbedLayer(
			embedSize, timeSteps, hiddenStates, categories, lrate, dictSize
		);
		gateLayer::OutputGate* layerOg = new gateLayer::OutputGate(
			embedSize, timeSteps, hiddenStates, categories, lrate
		);
		gateLayer::InputGate* layerIg = new gateLayer::InputGate(
			embedSize, timeSteps, hiddenStates, categories, lrate
		);
		gateLayer::ForgetGate* layerFg = new gateLayer::ForgetGate(
			embedSize, timeSteps, hiddenStates, categories, lrate
		);
		gateLayer::CellTGate* layerCt = new gateLayer::CellTGate(
			embedSize, timeSteps, hiddenStates, categories, lrate
		);
		gateLayer::CellGate* layerC = new gateLayer::CellGate(
			embedSize, timeSteps, hiddenStates, categories, lrate
		);
		gateLayer::HiddenGate* layerHg = new gateLayer::HiddenGate(
			embedSize, timeSteps, hiddenStates, categories, lrate
		);
		denseLayer::DenseLayer* layerDen = new denseLayer::DenseLayer(
			embedSize, timeSteps, hiddenStates, categories, lrate
		);

		std::unordered_map<std::string, basicLayer::BasicLayer*> layers{
		{"emb", layerEmbed}, {"og", layerOg}, {"ig", layerIg}, {"fg", layerFg},
		{"ct",layerCt}, {"c", layerC}, {"hg",layerHg}, {"den", layerDen} };

		//Load all trainSet, may need to modify
		std::vector<std::vector<int>>& trainX = ds->trainX;
		std::vector<int>& trainY = ds->trainY;

		std::cout << "Training Start!" << "\n";
		for (int ep = 0; ep < epoch; ep++) {
			float loss = 0;
			for (int sample = 0; sample < trainY.size(); sample++) {
				std::vector<int> x = trainX[sample];
				int y = trainY[sample];
				float* t = new float[3];
				t[y] = 1.0f;
				float* pred = forward(layers, x);
				loss = backward(pred, t, x, layers);
				std::cout << "\r" << "Sample: " << sample << "/" <<  std::flush;
			}
			std::cout << "Loss for epoch " << ep << "is :" << loss << std::endl;
		}
	}
	void LSTMNetwork::test(){}

	/********************************Private Methods******************************************/
	float* LSTMNetwork::forward(std::unordered_map<std::string, basicLayer::BasicLayer*>&
		layers, std::vector<int> x) {
		for (int t = 0; t < timeSteps; t++) {
			int textCode = x[t];
			float* embed = layers["emb"]->forward(textCode);
			float* h = oD->hs[t];//actually last h
			//followed by fg, ig, og, ct, c, h update
			oD->fgs[t + 1] = layers["fg"]->forward(embed, h, util::sigmoid);
			oD->igs[t + 1] = layers["ig"]->forward(embed, h, util::sigmoid);
			oD->ogs[t + 1] = layers["og"]->forward(embed, h, util::sigmoid);
			oD->cts[t + 1] = layers["ct"]->forward(embed, h, util::sigmoid);
			oD->cgs[t + 1] = layers["c"]->forward(oD);
			oD->hs[t + 1] = layers["hg"]->forward(oD);
		}
		return layers["den"]->forward(nullptr, oD->hs[timeSteps], util::softmax);
	}

	float LSTMNetwork::backward(float* pred, float* t, std::vector<int> x,
		std::unordered_map<std::string, basicLayer::BasicLayer*>& layers) {

		//dense
		float* h = oD->hs[timeSteps];
		float* dh = layers["den"]->backward(h, pred, t);
		float loss = layers["den"]->calLoss(pred, t);
		int emb = embedSize, hid = hiddenStates;

		//LSTM
		oD->dhs[timeSteps] = dh;
		for (int t = timeSteps; t > 0; t--) {
			//caldeltafollowed by og, fg,ig,ct
			float* embed = layers["emb"]->forward(x[t]);
			layers["og"]->calDeltak(oD, t);
			layers["fg"]->calDeltak(oD, t);
			layers["ig"]->calDeltak(oD, t);
			layers["ct"]->calDeltak(oD, t);
			oD->dhs[t - 1] = util::matElem(util::matElem(util::matElem(
				util::matMul(oD->dos[t], layers["og"]->getWh(), 1, emb, hid),
				util::matMul(oD->dis[t], layers["ig"]->getWh(), 1, emb, hid), 1, hid, '+'),
				util::matMul(oD->dfs[t], layers["fg"]->getWh(), 1, emb, hid), 1, hid, '+'),
				util::matMul(oD->dcs[t], layers["ct"]->getWh(), 1, emb, hid), 1, hid, '+');
		}
		float* xlast = layers["emb"]->forward(x.back());
		layers["og"]->calGrad(xlast, oD, &oD->dos);
		layers["fg"]->calGrad(xlast, oD, &oD->dfs);
		layers["ig"]->calGrad(xlast, oD, &oD->dis);
		layers["ct"]->calGrad(xlast, oD, &oD->dcs);
		layers["og"]->updateWb();
		layers["fg"]->updateWb();
		layers["ig"]->updateWb();
		layers["ct"]->updateWb();
		
		//embed
		std::vector<float*> deltasEmbed = getDeltaEmbed(layers["fg"]->getWx(), 
			layers["ig"]->getWx(), layers["ct"]->getWx(), layers["og"]->getWx());

		for (int i = 0; i < timeSteps; i++) {
			float* deltaE = deltasEmbed[i];
			int textCode = x[i];
			layers["emb"]->calGrad(deltaE, textCode);
			layers["emb"]->updateWb();
		}
		return loss;
	}
	std::vector<float*> LSTMNetwork::getDeltaEmbed(float* Wfx, float* Wix,
		float* Wcx, float* Wox) {

		int senLen = timeSteps;
		std::vector<float*> deltasEmbed(senLen, nullptr);
		for (int i = 0; i < senLen; i++) {
			float* dF = util::matMul(oD->dfs[i], Wfx, 1, hiddenStates, embedSize);
			float* dI = util::matMul(oD->dis[i], Wix, 1, hiddenStates, embedSize);
			float* dC = util::matMul(oD->dcs[i], Wcx, 1, hiddenStates, embedSize);
			float* dO = util::matMul(oD->dos[i], Wox, 1, hiddenStates, embedSize);
			float* d = util::matElem(util::matElem(util::matElem(dF,
				dI, 1, embedSize, '+'),
				dC, 1, embedSize, '+'),
				dO, 1, embedSize, '+');
			deltasEmbed[i] = d;
		}
		return deltasEmbed;
	}

	void LSTMNetwork::OutputsDeltaInit() {
		std::vector<float*> fgs(timeSteps + 1, new float[hiddenStates]());//may add 1
		std::vector<float*> igs(timeSteps + 1, new float[hiddenStates]());
		std::vector<float*> ogs(timeSteps + 1, new float[hiddenStates]());
		std::vector<float*> cgs(timeSteps + 1, new float[hiddenStates]());
		std::vector<float*> cts(timeSteps + 1, new float[hiddenStates]());
		std::vector<float*> hs(timeSteps + 1, new float[hiddenStates]());
		oD->fgs = fgs;
		oD->igs = igs;
		oD->ogs = ogs;
		oD->cgs = cgs;
		oD->cts = cts;
		oD->hs = hs;

		std::vector<float*> dfs(timeSteps + 1, new float[hiddenStates]());//may add 1
		std::vector<float*> dis(timeSteps + 1, new float[hiddenStates]());
		std::vector<float*> dos(timeSteps + 1, new float[hiddenStates]());
		std::vector<float*> dcs(timeSteps + 1, new float[hiddenStates]());
		std::vector<float*> dhs(timeSteps + 1, new float[hiddenStates]());
		oD->dfs = dfs;
		oD->dis = dis;
		oD->dos = dos;
		oD->dcs = dcs;
		oD->dhs = dhs;
	}
}