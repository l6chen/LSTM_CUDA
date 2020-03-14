# LSTM_CUDA
Implement LSTM purely with CUDA used to analyze sentiment

## TODO
1. Select Dataset for sentiment analysis
2. Specify which packages we need
3. Specify which parts of LSTM should be accelerated with CUDA 
4. Write python scipt to compare results

## Datasets
https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment

## Updates
1. Finished GateLayer Forward Algorithm
2. Know that four gates are objects of class GateLayer. The GateLayer and DenseLayer should be child of Basic Layer.
3. Need to understand the step of backward update.
