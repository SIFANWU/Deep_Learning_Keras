# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:41:31 2018

@author: Jim Wu
"""

from keras.datasets import boston_housing
from keras import models
from keras import layers

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data-=mean
std = train_data.std(axis =0)
train_data/=std
'''
axis = 0 代表列 axis = 1 代表行
'''
test_data-=mean
test_data/=std

def build_model():
    '''
    这是标量回归，最后一层是纯线性的，网络可以学会预测任意范围内的值。
    loss = mse (Mean Squared Error) 这是回归问题的常用loss function
    metics= mae (Mean absolute Error) 预测与目标的绝对差值
    optimizer = rmsprop 在deep learning 都是一个不错的选择
    '''
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

'''
############################################################################
                                K折交叉验证法 
            专门应对数据量较少，导致验证集的验证效果不好的一种方法。
            
Description:
                这种方法将可用数据划分为K个分区(K值通常取4或者5)
                实例化K个相同的模型，将每个模型在K-1个分区上训练,
                并在剩下的一个分区上进行评估.模型的验证分数等于
                K个验证分数的平均值。

Graphic:
                              三个分区
    第1折  第一分区(验证)    第二分区          第三分区       '第一个验证分数'
    第2折  第一分区          第二分区(验证)    第三分区       '第二个验证分数'
    第3折  第一分区          第二分区          第三分区(验证) '第三个验证分数'
    
                        最终分数:取三个验证分数的平均数

############################################################################
'''

import numpy as np
import matplotlib.pyplot as plt

k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i  in range(k):
    print('Processing Fold ', i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate([train_data[:i*num_val_samples],
                                         train_data[(i+1)*num_val_samples:]],
                                        axis=0)
    
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],
                                         train_targets[(i+1)*num_val_samples:]],
                                        axis=0)    
    model =build_model()
    history = model.fit(partial_train_data,partial_train_targets,
              validation_data=(val_data,val_targets),
              epochs=num_epochs,
              batch_size=1) # verbose=0静默模式
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
    average_mae_history = [ np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

'''
######################################################
数据方差相对较大，很难看清规律
删除前10个数据点，因为它们的取值范围与曲线上的其他点不同
将每个数据点替换为前面数据点的指数移动平均值，以得到光滑的曲线
######################################################
'''    
def smooth_curve(points,factor=0.9):
    smoothed_pointers = []
    for point in points:
        if smoothed_pointers:
            previous = smoothed_pointers[-1]
            smoothed_pointers.append(previous*factor + point*(1-factor))
        else:
            smoothed_pointers.append(point)
    return smoothed_pointers

smooth_mae_history = smooth_curve(average_mae_history[10:])
            
plt.plot(range(1,len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
    
'''
#从图上看出 验证MAE在80轮后不再显著降低，之后就开始过拟合
#所以最终模型跑80次就可以了
model = build_model()
model.fit(train_data,train_targets,
          epochs= 80,batch_size=16,verbose=0)
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
'''


    