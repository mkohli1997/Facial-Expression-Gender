
import cv2
from keras.models import load_model
import os
import numpy as np

xAnger=[]
xDisgust=[]
xFear=[]
xHappy=[]
xNeutral=[]
xSad=[]
xSurprise=[]

xAll = [xAnger, xFear, xHappy, xNeutral, xSad, xSurprise]

print('Loading Angry dataset...')
for image in os.listdir('emotion_data/angry/'):
    img=cv2.imread('emotion_data/angry/%s'%image,0)
    img=cv2.resize(img,(64,64))
    img=np.reshape(img,(1,64,64,1))
    xAnger.append(img)

#print('Loading Disgust dataset...')
#for image in os.listdir('emotion_data/disgust/'):
     #img=cv2.imread('emotion_data/disgust/%s'%image,0)
     #img=cv2.resize(img,(64,64))
     #img=np.reshape(img,(1,64,64,1))
     #xDisgust.append(img)
    

print('Loading Fear dataset...')
for image in os.listdir('emotion_data/fear/'):
    img=cv2.imread('emotion_data/fear/%s'%image,0)
    img=cv2.resize(img,(64,64))
    img=np.reshape(img,(1,64,64,1))
    xFear.append(img)

print('Loading Happy dataset...')
for image in os.listdir('emotion_data/happy/'):
    img=cv2.imread('emotion_data/happy/%s'%image,0)
    img=cv2.resize(img,(64,64))
    img=np.reshape(img,(1,64,64,1))
    xHappy.append(img)

print('Loading Neutral dataset...')
for image in os.listdir('emotion_data/neutral/'):
    img=cv2.imread('emotion_data/neutral/%s'%image,0)
    img=cv2.resize(img,(64,64))
    img=np.reshape(img,(1,64,64,1))
    xNeutral.append(img)

print('Loading Sad dataset...')
for image in os.listdir('emotion_data/sad/'):
    img=cv2.imread('emotion_data/sad/%s'%image,0)
    img=cv2.resize(img,(64,64))
    img=np.reshape(img,(1,64,64,1))
    xSad.append(img)

print('Loading Surprise dataset...')
for image in os.listdir('emotion_data/surprise/'):
    img=cv2.imread('emotion_data/surprise/%s'%image,0)
    img=cv2.resize(img,(64,64))
    img=np.reshape(img,(1,64,64,1))
    xSurprise.append(img)



#5 for all
def createMatrix():
    
    emotion_model = model_e=load_model('saved_weights_emotion/weights-73-1.031777-0.613.hdf5')
    
    confusionMatrix = [[0]*6 for _ in range(6)]

    for known in range(len(xAll)):
        for image in xAll[known]:
            prediction_emo=emotion_model.predict(image/255.0)
            classification = np.argmax(prediction_emo)
            confusionMatrix[classification][known] += 1
    
    print(confusionMatrix)

#createMatrix()
print(len(xAnger))
print(len(xFear))
print(len(xHappy))
print(len(xNeutral))
print(len(xSad))
print(len(xSurprise))