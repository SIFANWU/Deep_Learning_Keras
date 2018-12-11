# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:43:59 2018

@author: Administrator
"""
'''
##########################################
                K折验证
在最终训练之前，用于验证集，避免过拟合问题。
##########################################
'''
import numpy as np 
from keras import models
from keras import layers

k = 4 #can be anyone (it is usually between 3 and 9 )
num_validation_samples = len(data)//k

np.random.shuffle(data) # disrupt array order

validation_scores=[]
for fold in range(k):
    validation_data = data [num_validation_samples*fold : num_validation_samples*(fold+1)]
    train_data = data [:num_validation_samples*fold] + data [num_validation_samples*(fold+1):]
    
    model = getmodel()# get  the model
    model.train(train_data) # training step
    validation_score = model.evaluate(validation_data) # get one score of each fold
    validation_scores.append(validation_score) 

average_validation_scores = np.average(validation_scores) # calcualte the average of validation scores


def getmodel():
    '''
    Build the model struction.
    '''
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])    
    
    return model
    
    




















