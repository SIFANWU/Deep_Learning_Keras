# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:46:38 2018

@author: Administrator
"""

from keras import models,optimizers,layers,losses,metrics
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
 

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
'''
#解码 将索引转换为评论 
word_index =imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])
decoded_review = ''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
#print(decoded_review)
#return 'thisfilmwasjustbrilliantcastinglocationscenerystorydirectioneveryone'
'''

def vectorize_sequences(sequences,dimension=10000):
    '''
    将整数序列编码为二进制矩阵
    '''
    results = np.zeros((len(sequences),dimension))
    for i, sequences in enumerate(sequences):
        results[i,sequences]=1.0
    return results

x_train = vectorize_sequences(train_data)#将训练数据向量化
x_test =  vectorize_sequences(test_data)#将测试数据向量化

y_train = np.asarray(train_labels).astype('float32')#标签向量化
y_test = np.asarray(test_labels).astype('float32')#标签向量化

model = models.Sequential()
model.add(layers.Dense(16,activation ='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation= 'relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

#留出10 000 个样本 作为验证集
x_value = x_train[:10000]
partial_x_value = x_train[10000:]

y_value = y_train[:10000]
partial_y_value = y_train[10000:]

#人为监控过拟合的发生，使用验证集合(从训练数据中筛选出一部分用作验证)
history = model.fit(partial_x_value,partial_y_value,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_value,y_value))

results = model.evaluate(x_test,y_test)
print('\n测试集结果：',results)


history_dict=history.history
#print(history_dict.keys())
#return dict_keys(['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy'])

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label='Training loss')#'bo'表示蓝色圆点
plt.plot(epochs,val_loss_values,'b',label='Validation loss')#'b'表示蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()#显示画线的信息

plt.show()

plt.clf()#清空图像
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


