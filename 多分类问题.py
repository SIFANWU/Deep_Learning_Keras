# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:57:44 2018

@author: Administrator
"""
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models,layers
import matplotlib.pyplot as plt

(train_data,train_labels),(test_data,test_labels) =reuters.load_data(num_words=10000)

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequences in enumerate(sequences):
        results[i,sequences]=1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))

#print(history.history.keys())
#dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs= range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()#清空图像
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


#---------------------------------------#
'''
从2个图中看出网络在训练9轮后开始过拟合
从头开始训练一个新网络，共9轮
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val,y_val))

results = model.evaluate(x_test,one_hot_test_labels)


'''
#return results
#[0.956,0.796]
#---------------------------------------#

predictions= model.predict(x_test)

#>>>predictions[0].shape
# (46,)

#>>>np.sum(predictions[0])
# 1.0

#>>>np.argmax(predictions[0])
# 4








