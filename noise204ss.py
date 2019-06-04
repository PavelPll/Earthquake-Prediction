#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Predict the time (single value) remaining before laboratory earthquakes occur 
#from real-time seismic data (150000 values) 


# In[ ]:


import numpy as np 
import pandas as pd 
import os
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


# In[ ]:


#read data
float_data = pd.read_csv("train.csv", #nrows=2e100, 
                         dtype={"acoustic_data": np.float32, 
                                "time_to_failure": np.float32})
float_data = float_data.values #np array


# In[ ]:


#divide the segment of 150000 consecutive values into 75 smaller segments
N_STEPS=75
STEP_LENGTH=2000
DEL=8
print("last: ",int(round(STEP_LENGTH/DEL)))
print("last: ",int(round(STEP_LENGTH/(DEL*DEL))))
#SPE is step per epoch, each epochs is trained on 32*1000 segments
SPE=1000


# In[ ]:


#some functions to create augmenters
from scipy.signal import savgol_filter
import pywt
from skimage.restoration import (denoise_wavelet, estimate_sigma)

def running_mean(x, N=3):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    result=(cumsum[N:] - cumsum[:-N]) / float(N)
    result=np.insert(result, 0, x[0])
    result=np.append(result,x[len(x)-1])
    return np.matrix.round(result,0)

def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# In[ ]:


#the idea: convert 150000 values into features to decrease the number of values for RNN input
from scipy.stats import normaltest
from scipy.stats import moment, kurtosis, skew
from tsfresh.feature_extraction import feature_calculators as ts
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from random import randint

def extract_features(z):
    #print(z.shape)
    #z = z + np.random.normal(0, 0.5, [z.shape[0],z.shape[1]])
    #b = normaltest(z, axis=1)
    #m3 = np.cbrt( moment(z, 3, axis=1) )
    
    #m21 = autocorr1(z,[1])
    #print("m21check ", m21>0)
    #print("mean ",z.mean(axis=1).shape)
    #print("m21 ",m21.shape)
    return np.c_[z.mean(axis=1), 
                  np.median(np.abs(z), axis=1),
                  z.std(axis=1), 
                  z.max(axis=1),
                  z.min(axis=1),
                  #kurtosis(z, axis=1),
                  #-skew(z, axis=1),
                  np.quantile(np.abs(z), 0.05, axis=1),
                  np.quantile(np.abs(z), 0.25, axis=1),
                  np.quantile(np.abs(z), 0.75, axis=1),
                  np.quantile(np.abs(z), 0.95, axis=1),
                  #1-np.quantile(z, 0.75, axis=1),
                  #b[1],
                  #-m3,
                  #m21,
                  #z.shape[1]
                ]

# For a given ending position "last_index", we split the last 150'000 values of "x" into 150 pieces of length 1000 each.
# From each piece, 16 features are extracted. This results in a feature matrix of dimension (150 time steps x 16 features). 
def create_X(x, last_index=None, n_steps=N_STEPS, step_length=STEP_LENGTH, aug=0):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    per=x[(last_index - n_steps * step_length):last_index]
    #print("per", x.shape)

    #for data augmentation
    if aug==1:
        flag=randint(0, 3)
        if flag==0:
            s=np.random.normal(0, 1, per.shape[0])
            s=np.matrix.round(s,0)
            per=per+s
        if flag==1:
            per=running_mean(per)
        if flag==2:
            per=savgol_filter(per, 5, polyorder=3)
            per=np.matrix.round(per,0)
        if flag==3:
            per=lowpassfilter(per, thresh = 0.01, wavelet="db4")
            per=np.matrix.round(per,0)
    
    temp = (per.reshape(n_steps, -1) - 5 ) / 3
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    q05_roll_std_10=np.zeros(n_steps)
    R_std=np.zeros(n_steps)
    av_change_abs_roll_mean_10=np.zeros(n_steps)
    Imean=np.zeros(n_steps)

    
    
    mac=np.zeros(n_steps)
    mc=np.zeros(n_steps)
    
    for i in range(n_steps):
        #s=pd.DataFrame(temp[i, :])
        #x_roll_std=s.rolling(10).std().dropna().values
        x_roll_std=rolling_window(temp[i,:], 10).std(axis=1)
        #print("rol ", x_roll_std.shape)
        q05_roll_std_10[i]=np.quantile(x_roll_std, 0.05)
        
        zc=np.fft.fft(temp[i, :])
        realFFT=np.real(zc)
        R_std[i]=realFFT.std()
        imagFFT=np.imag(zc)
        Imean[i]=imagFFT.mean()
        
        #x_roll_mean = s.rolling(10).mean().dropna().values
        x_roll_mean=rolling_window(temp[i,:], 10).mean(axis=1)
        av_change_abs_roll_mean_10[i] = np.mean(np.diff(x_roll_mean))        
        
        
        mac[i]=ts.mean_abs_change(temp[i,:])
        mc[i]=ts.mean_change(temp[i,:])
        
    return np.c_[extract_features(temp),
                 extract_features(temp[:, ( step_length-int(round(step_length/DEL)) ):]),
                 extract_features(temp[:, ( step_length-int(round(step_length/(DEL*DEL))) ):]),
                 q05_roll_std_10,
                 R_std,
                 av_change_abs_roll_mean_10,
                 Imean,
                 mac,
                 mc,
                 temp[:, -1:]]

# We call "extract_features" three times, so the total number of features is 9 * 3 + 7 (last value) = 34


# In[ ]:


#to provide the same input for all epochs based on single random sampling of the segments 
#with 150000 length
#It is achieved by adding new generator 

batch_size = 64
batch_size1=int(batch_size/2)

min_index=0
max_index = int(len(float_data) - 1)

np.random.seed(seed=1)
arr_rows=[]
for i in range(SPE):
    rows = np.random.randint(min_index + N_STEPS * STEP_LENGTH, max_index, size=batch_size1)
    arr_rows.append(rows)
    
def gf(min_index, n_steps, step_length, batch_size1):
    while True:
        #np.random.seed(seed=1)
        i=0
        while i<SPE*(1):
            #rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size1)
            rows=arr_rows[i]
            yield rows 
            i=i+1
gen = gf(min_index, N_STEPS, STEP_LENGTH, int(batch_size/2))


# In[ ]:


#generate input for RNN for real data + augmentation
n_features = create_X(float_data[0:STEP_LENGTH*N_STEPS,0], 
                      n_steps=N_STEPS, step_length=STEP_LENGTH).shape[1] 
print("n_features= ",n_features)
    
# The generator randomly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_X".
def generator(data, min_index=0, max_index=None, batch_size=32, n_steps=N_STEPS, 
              step_length=STEP_LENGTH, val=0):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        if val==0:
            batch_size1=int(batch_size/2)
            rows=next(gen)
        else:
            batch_size1=batch_size
            rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size1)
        #np.random.seed(seed=1)
        #rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size1)
        #rows=next(gen)
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        
        delta=len(rows)
        for j, row in enumerate(rows):
            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, 
                                  step_length=step_length)
            targets[j] = data[row, 1]
            if val==0:
                samples[j+delta] = create_X(data[:, 0], last_index=row, n_steps=n_steps, 
                                            step_length=step_length, aug=1)
                targets[j+delta] = data[row, 1]
#         if val==0:
#             print(targets[0])
        yield samples, targets


# In[ ]:


train_gen = generator(float_data, batch_size=batch_size, val=0)
#no data augmentation for validation
valid_gen = generator(float_data, batch_size=batch_size, val=1)


# In[ ]:


# Define model
import keras
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, Dropout, GRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint


# In[ ]:


from keras import backend
print(backend.tensorflow_backend._get_available_gpus())


# In[ ]:


# Define model
cb = ModelCheckpoint("model.hdf5", monitor='val_loss', save_weights_only=False, period=1)

model = Sequential()
#model.add(GRU(100, return_sequences=True, input_shape=(None, n_features)))
model.add(GRU(68, input_shape=(None, n_features)))
#model.add(GRU(21))
model.add(Dense(20, activation='relu'))
#model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1))

model.summary()
model.compile(optimizer=adam(lr=0.0005), loss="mae")


# In[ ]:


#define learning rate
from os import rename
from os.path import isfile

def lr_schedule(epoch):
    #arr=np.array([0.1e-5, 0.1e-4, 0.1e-3, 0.1e-3])
    #lr=arr[epoch]
#     if epoch<0:
#         lr=0.05e-4
#     else:
#         lr = 0.5e-04
    lr=0.0005
    print('Learning rate: ', lr)
    
    if isfile("model.hdf5"):
        n="model_noise204_ep"+str(epoch)+".hdf5"
        rename("model.hdf5",n)
        print("renamed to ",n)
    else:
        print("no file to rename")
        
        
    return lr
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)


# In[ ]:


history = model.fit_generator(train_gen,
                              steps_per_epoch=SPE,#n_train // batch_size,
                              epochs=200,
                              verbose=2,
                              #callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=100,
                              callbacks=[cb, lr_scheduler])#n_valid // batch_size)
#val_loss calculation is based on random sampling (32*100 pieces of 150000 consecutive values from 6e6 values)
#this is the way to evaluate the model on the whole dataset
#overfitting is compensated by low number of parameters (22063<<150000) in the model 
#and by data augmentation


# In[ ]:


#generating submission file
from keras.models import load_model
bestModel = load_model('model_noise204_ep20.hdf5')

submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
from tqdm import tqdm_notebook
# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm_notebook(submission.index)):
  #  print(i)
    seg = pd.read_csv('test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = bestModel.predict(np.expand_dims(create_X(x), 0))

submission.head()

# Save
submission.to_csv('submission_noise204_ep20.csv')


# In[ ]:




