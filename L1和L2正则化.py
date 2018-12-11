# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:24:01 2018

@author: Administrator
"""

from keras import regularizers
from keras import models
from keras import layers

'''
regularizers.l1(0.001) # L1正则化
regularizers.l1_l2(l1=0.001,l2=0.001) # 同时做L1和L2正则化 
'''

model = models.Sequential()
model.add(layers.Dense(16,kernal_regularizer = regularizers.l2(0.001),
                       activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,kernal_regularizer = regularizers.l2(0.001),
                       activation='relu')
model.add(layers.Dense(1,activation='sigmoid'))

#l2(0.001)的意思是该层权重矩阵的每个系数都会使网络总损失增加0.001*weight_coefficient_value。
#注意，由于这个惩罚项只在训练时添加，所以这个网络的训练损失会比测试损失大很多。
'''
#######################################
        使用Dropout的方法
取值在0.2~0.5
#######################################
'''
model = models.Sequential()
model.add(layers.Dense(16, activation='relu',input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu')
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid')



