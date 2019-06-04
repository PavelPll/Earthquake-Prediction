# Earthquake-Prediction 
Kaggle project: Predict the time remaining before laboratory earthquakes occur from real-time seismic data. 
4541 Teams
https://www.kaggle.com/c/LANL-Earthquake-Prediction

I started from this kernel: https://www.kaggle.com/mayer79/rnn-starter-for-huge-time-series

My best submission: noise9ss.py Public score: 1.625 Private score: 2.501 Final standing between 236 and 237, i.e., top 5.2%
My final submission: noise204ss.py Public score: 1.473 Private score: 2.675 Final standing 3061, i.e. top 67%

As it can be seen the public score was not a measure of the model performance. 
In general, I think RNN (LSTM, GRU) is not the best way to solve this problem because the number of features is limited by computation resources. The best way is to use regression (Random Forest or LGBM) for significantly increased number of features. 


