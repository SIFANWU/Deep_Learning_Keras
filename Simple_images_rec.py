from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models

#get the train and test data 
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#build each layers for network
network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

#compile the network
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#reshape the size of input data and change type
#make sure the range of value [0~1] 
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255
#decode for images labels into classes
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#train the network
network.fit(train_images,train_labels,epochs=5,batch_size=128)
#run the test simples to get the accuracy
test_loss,test_acc=network.evaluate(test_images,test_labels)
print(test_loss,test_acc)

#display the number in mnist
''' 
digit = train_images[1]

import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary) #cmap: set the colour
plt.show()
'''