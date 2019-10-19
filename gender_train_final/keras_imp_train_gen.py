from keras.layers import Dense,Flatten,Conv2D,Dropout,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import *
from keras.utils import to_categorical,plot_model
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x=[]
y=[]

print('Loading Female dataset...')
for image in os.listdir('gender_data/female'):
    img=cv2.imread('gender_data/female/%s'%image,0)
    img=cv2.resize(img,(128,128))
    x.append(img)
    y.append(0)

print('Loading Male dataset...')
for image in os.listdir('gender_data/male'):
    img=cv2.imread('gender_data/male/%s'%image,0)
    img=cv2.resize(img,(128,128))
    x.append(img)
    y.append(1)

x=np.array(x)
x=x/255.0 # normalize numpy arrays' values
y=np.array(y)
x=x.reshape(x.shape[0],128,128,1)
y=to_categorical(y,2) #convert to categorical format
x, y = shuffle(x, y) #shuffle image-label pairs
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2) #train-test split 80%-20%

print('Total number of images = ', x.shape[0])
print(x_train.shape)
print(x_train[0].shape)
input_shape = x_train.shape[1:]
print(input_shape)
pool_size = (2, 2)
model=Sequential()
#

model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size))
#
model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))


model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))


#define learning rate and the learning rate decay per epoch
lr=1e-3
decay=1e-5

model.compile(Adam(lr=lr,decay=decay),loss='categorical_crossentropy',metrics=['accuracy'])

plot_model(model, to_file='gender_model_architecture.png',show_shapes=True)
# here we plot the model architecture

filepath="weights-{epoch:02d}-{val_loss:4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# here we save the weights when we observe minimum validation loss

model.summary()
history=model.fit(x_train,y_train,epochs=30,validation_data=(x_val, y_val),batch_size=128,verbose=2,callbacks=[checkpoint])

#plot our results

#summarize for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#

