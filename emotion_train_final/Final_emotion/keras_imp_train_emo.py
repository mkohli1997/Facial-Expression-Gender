import keras
from keras.layers import Dense,Flatten,InputLayer,Conv2D,Dropout,MaxPooling2D,UpSampling2D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import *
from keras.utils import to_categorical, plot_model
from keras.layers.normalization import BatchNormalization
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKeras

x=[]
y=[]

print('Loading Angry dataset...')
for image in os.listdir('emotion_data2/angry/'):
    img=cv2.imread('emotion_data2/angry/%s'%image,0)
    img=cv2.resize(img,(64,64))
    x.append(img)
    y.append(0)

# print('Loading Disgust dataset...')
# for image in os.listdir('D:\\Sem6\\Minor_Project\\Final\\EmotionClassification\\training_data\\disgust'):
#     img=cv2.imread('D:\\Sem6\\Minor_Project\\Final\\EmotionClassification\\training_data\\disgust\\%s'%image,0)
#     img=cv2.resize(img,(64,64))
#     x.append(img)
#     y.append(1)

print('Loading Fear dataset...')
for image in os.listdir('emotion_data2/fear/'):
    img=cv2.imread('emotion_data2/fear/%s'%image,0)
    img=cv2.resize(img,(64,64))
    x.append(img)
    y.append(1)

print('Loading Happy dataset...')
for image in os.listdir('emotion_data2/happy/'):
    img=cv2.imread('emotion_data2/happy/%s'%image,0)
    img=cv2.resize(img,(64,64))
    x.append(img)
    y.append(2)

print('Loading Neutral dataset...')
for image in os.listdir('emotion_data2/neutral/'):
    img=cv2.imread('emotion_data2/neutral/%s'%image,0)
    img=cv2.resize(img,(64,64))
    x.append(img)
    y.append(3)

print('Loading Sad dataset...')
for image in os.listdir('emotion_data2/sad/'):
    img=cv2.imread('emotion_data2/sad/%s'%image,0)
    img=cv2.resize(img,(64,64))
    x.append(img)
    y.append(4)

print('Loading Surprise dataset...')
for image in os.listdir('emotion_data2/surprise/'):
    img=cv2.imread('emotion_data2/surprise/%s'%image,0)
    img=cv2.resize(img,(64,64))
    x.append(img)
    y.append(5)

x=np.array(x)
print(x.shape)
x=x/255.0 #normalizing the numpy array before feeding into the network
y=np.array(y)
x=x.reshape(x.shape[0],64,64,1)
y=to_categorical(y,6) #convert to categorical format

x, y = shuffle(x, y)#shuffle the image-label pairs
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25) # train-test split of 75%-25%

print(y.shape)
print(x_train.shape)
print(x_train[0].shape)
input_shape = x_train.shape[1:]
print(input_shape)

pool_size = (2, 2)

model=Sequential()

model.add(Conv2D(32,(5,5),strides=(1,1),padding='same',activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Conv2D(64,(5,5),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))

model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(6,activation='softmax'))


#
lr=1e-4 #define learning rate
decay=1e-6 # define decay rate - slows down the learning rate with each epoch
model.compile(Adam(lr=lr,decay=decay),loss='categorical_crossentropy',metrics=['accuracy'])

filepath="saved_weights_emotion/weights-{epoch:02d}-{val_loss:4f}-{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# here we save the weights when we observe minimum validation loss

model.summary()
plot_model(model,to_file='emotion_architecture.png',show_shapes=True)
history=model.fit(x_train,y_train,epochs=100,validation_data=(x_val, y_val),batch_size=256,verbose=2,callbacks=[checkpoint])


#plotting our results
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

