# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:13:11 2018

@author: Administrator
"""

'''
####################################
            原始模型
####################################
'''
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu')) # activation function random selection
model.add(layers.Dense(1,activation='sigmoid')) # activation function random selection

'''
####################################
        容量更小的模型
####################################
'''
model = models.Sequential()
model.add(layers.Dense(4,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(4,activation='relu')) # activation function random selection
model.add(layers.Dense(1,activation='sigmoid')) # activation function random selection
#更小的模型开始过拟合的时间较晚，过拟合后性能变差的速度也更慢。
'''
####################################
        容量更大的模型
####################################
'''
model = models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(512,activation='relu')) # activation function random selection
model.add(layers.Dense(1,activation='sigmoid')) # activation function random selection
#更大的模型只过了极少的轮数后开始过拟合，过拟合也更为严重。








