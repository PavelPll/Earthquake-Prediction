#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Predict the time (single vlue) remaining before laboratory earthquakes occur 
#from real-time seismic data (150000 values) 


# In[2]:


import numpy as np 
import pandas as pd 
import os
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


# In[3]:


#read data
float_data = pd.read_csv("train.csv", #nrows=2e100, 
                         dtype={"acoustic_data": np.float32, 
                                "time_to_failure": np.float32})
float_data = float_data.values #np array


# In[6]:


#two functions to create augmenters

def running_mean(x, N=3):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    result=(cumsum[N:] - cumsum[:-N]) / float(N)
    result=np.insert(result, 0, x[0])
    result=np.append(result,x[len(x)-1])
    return np.matrix.round(result,0)

#frequency filter
def fourier(x):
    y1=np.fft.fft(x)
    l=len(y1)
    freq = np.fft.fftfreq(len(y1))
    
    mean=np.mean(np.abs(y1))
    seuil=mean+np.std(np.abs(y1)-mean)

    obrez=np.floor((3/5)*l/2)
    #print("obrez", obrez)
    for i in range(len(y1)):
    #if (i<np.floor(l*0.1))|(i>np.floor(0.8*l)):
        if ((i>obrez) and i<(l-obrez)):
            #if np.abs(y1[i])<seuil:
                y1[i]=0
    #plt.plot(np.abs(y1))   
    #inverse fft to recostruct the signal
    yi=np.fft.ifft(y1)
    return np.matrix.round(np.real(yi),0)


# In[7]:


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
    m3 = np.cbrt( moment(z, 3, axis=1) )
    
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
                  -skew(z, axis=1),
                  np.quantile(np.abs(z), 0.25, axis=1),
                  np.quantile(np.abs(z), 0.75, axis=1),
                  #1-np.quantile(z, 0.75, axis=1),
                  #b[1],
                  -m3,
                  #m21,
                  #z.shape[1]
                ]

# For a given ending position "last_index", we split the last 150'000 values of "x" into 150 pieces of length 1000 each.
# From each piece, 34 features are extracted. This results in a feature matrix of dimension (150 time steps x 34 features). 
def create_X(x, last_index=None, n_steps=150, step_length=1000, aug=0):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0

    # Reshaping
    per=x[(last_index - n_steps * step_length):last_index]

    #for data augmentation
    if aug==1:
        flag=randint(0, 2)
        if flag==0:
            s=np.random.normal(0, 1, per.shape[0])
            s=np.matrix.round(s,0)
            per=per+s
        if flag==1:
            per=running_mean(per)
        if flag==2:
            per=fourier(per)
            #print(per)

    temp = (per.reshape(n_steps, -1) - 5 ) / 3
    
    #ac1=np.zeros(150)
    ac2=np.zeros(150)
    ac3=np.zeros(150)
    #c3_1=np.zeros(150)
    c3_2=np.zeros(150)
    c3_3=np.zeros(150)
    mac=np.zeros(150)
    mc=np.zeros(150)
    for i in range(150):
        #ac1[i]=ts.autocorrelation(temp[i,:],1)
        ac2[i]=ts.autocorrelation(temp[i,:],2)
        ac3[i]=ts.autocorrelation(temp[i,:],3)
        #c3_1[i]=ts.c3(temp[i,:],1)/500
        c3_2[i]=ts.c3(temp[i,:],2)/500
        c3_3[i]=ts.c3(temp[i,:],3)/500
        mac[i]=ts.mean_abs_change(temp[i,:])
        mc[i]=ts.mean_change(temp[i,:])
        
    return np.c_[extract_features(temp),
                 extract_features(temp[:, 827:]),
                 extract_features(temp[:, 970:]),
                 #ac1,
                 ac2,
                 ac3,
                 #c3_1,
                 c3_2,
                 c3_3,
                 mac,
                 mc,
                 temp[:, -1:]]

# We call "extract_features" three times, so the total number of features is 9 * 3 + 7 (last value) = 34


# In[9]:


#generate input for RNN for real data + augmentation
n_features = create_X(float_data[0:150000,0], n_steps=150, step_length=1000).shape[1] 
print("n_features= ",n_features)
    
# The generator randomly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_X".
#Non-random selection gives worse result
def generator(data, min_index=0, max_index=None, batch_size=32, n_steps=150, step_length=1000, val=0):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        if val==0:
            #the first half is real data
            #the second half is related to augmented data
            batch_size1=int(batch_size/2)
        else:
            # no augmentation for data validation
            batch_size1=batch_size
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size1)
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        
        delta=len(rows)
        for j, row in enumerate(rows):
            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
            targets[j] = data[row, 1]
            #add data augmentation
            if val==0:
                samples[j+delta] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length, aug=1)
                targets[j+delta] = data[row, 1]
        yield samples, targets


# In[ ]:


#to provide the same input for all epochs based on single random samling of the segments with 150000 length
#It is achieved by adding new generator 
#This part was removed because it does not allow to improve the result

# N_STEPS=150
# STEP_LENGTH=1000
# DEL=5.8
# print("last: ",int(round(STEP_LENGTH/DEL)))
# print("last: ",int(round(STEP_LENGTH/(DEL*DEL))))
# #define number of batches per epoch
# SPE=1000

# batch_size = 64
# #for 50% augmentation
# batch_size1=int(batch_size/2)

# min_index=0
# max_index = int(len(float_data) - 1)

# np.random.seed(seed=1)
# arr_rows=[]
# for i in range(SPE):
#     rows = np.random.randint(min_index + N_STEPS * STEP_LENGTH, max_index, size=batch_size1)
#     arr_rows.append(rows)
    
# def gf(min_index, n_steps, step_length, batch_size1):
#     while True:
#         np.random.seed(seed=1)
#         i=0
#         while i<SPE*(1):
#             #rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size1)
#             rows=arr_rows[i]
#             yield rows 
#             i=i+1
# gen = gf(min_index, N_STEPS, STEP_LENGTH, int(batch_size/2))


# In[10]:


batch_size = 64

train_gen = generator(float_data, batch_size=batch_size, val=0)
#remove augmentation for validation
valid_gen = generator(float_data, batch_size=batch_size, val=1)


# In[12]:


import keras
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, Dropout, GRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint


# In[13]:


from keras import backend
print(backend.tensorflow_backend._get_available_gpus())


# In[14]:


# Define model
cb = ModelCheckpoint("model.hdf5", monitor='val_loss', save_weights_only=False, period=1)

model = Sequential()
#model.add(GRU(100, return_sequences=True, input_shape=(None, n_features)))
model.add(GRU(68, input_shape=(None, n_features)))
#model.add(GRU(21))
model.add(Dense(15, activation='relu'))
#model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1))

model.summary()
model.compile(optimizer=adam(lr=0.0005), loss="mae")


# In[15]:


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
        n="model_noise9_ep"+str(epoch)+".hdf5"
        rename("model.hdf5",n)
        print("renamed to ",n)
    else:
        print("no file to rename")
        
    return lr
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)


# In[16]:


history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,#n_train // batch_size,
                              epochs=200,
                              verbose=2,
                              #callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=100,
                              callbacks=[cb, lr_scheduler])#n_valid // batch_size)
#val_loss calculation is based on random sampling (32*1000 pieces of 150000 consecutive values from 6e6 values)
#this is the way to evaluate the model on the whole dataset
#overfitting is compensated by low number of parameters (22063<<150000) in the model 
#and by data augmentation


# In[19]:


#generating submission
from keras.models import load_model
bestModel = load_model('model_noise9_ep14.hdf5')


# In[20]:


submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
from tqdm import tqdm
# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = bestModel.predict(np.expand_dims(create_X(x), 0))

submission.head()

# Save
submission.to_csv('submission_noise9_ep14.csv')


# In[ ]:




