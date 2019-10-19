import cv2
from keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_g=load_model('saved_weights_gender/weights-17-0.134326-0.950.hdf5')
model_e=load_model('saved_weights_emotion/weights-33-1.103463-0.601.hdf5')

def pred_gender(roi):

    roi=cv2.resize(roi,(128,128))
    roi=np.reshape(roi,(1,128,128,1))
    res=model_g.predict(roi)
    return res

def pred_emotion(roi):

    roi=cv2.resize(roi,(64,64))
    roi=np.reshape(roi,(1,64,64,1))
    res=model_e.predict(roi)
    return res


def getFrame():
    cap=cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret,frame=cap.read()
        img=np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img,1.3,5)

        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            prediction_gen = pred_gender(roi/255.0)
            prediction_emo=pred_emotion(roi/255.0)
            pos_g=np.argmax(prediction_gen)
            pos_e=np.argmax(prediction_emo)
            if pos_g==0:
                gender='FEMALE'
            else:
                gender='MALE'
            font = cv2.FONT_HERSHEY_SIMPLEX
            if (gender == 'MALE'):
                cv2.putText(img, gender, (x, y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: '+'%.4f'%prediction_gen[0][1],(x+w,y), font, 0.75, (0,255,255),2,cv2.LINE_AA)
            elif (gender == 'FEMALE'):
                cv2.putText(img, gender, (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: '+'%.4f'%prediction_gen[0][0],(x+w,y), font, 0.75, (0,255,255),2,cv2.LINE_AA)

            if pos_e==0:
                emotion='ANGRY'
            elif pos_e==1:
                emotion='DISGUST'
            elif pos_e==2:
                emotion='FEAR'
            elif pos_e==3:
                emotion='HAPPY'
            elif pos_e==4:
                emotion='NEUTRAL'
            elif pos_e==5:
                emotion='SAD'
            elif pos_e==6:
                emotion='SURPRISE'

            if (emotion == 'ANGRY'):
                cv2.putText(img, emotion, (x, y+h), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: '+'%.4f'%prediction_emo[0][0],(x+w,y+h), font, 0.75, (0,255,255),2,cv2.LINE_AA)
            elif (emotion == 'DISGUST'):
                cv2.putText(img, emotion, (x, y+h), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: '+'%.4f'%prediction_emo[0][1],(x+w,y+h), font, 0.75, (0,255,255),2,cv2.LINE_AA)
            elif (emotion == 'FEAR'):
                cv2.putText(img, emotion, (x, y+h), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: '+'%.4f'%prediction_emo[0][2],(x+w,y+h), font, 0.75, (0,255,255),2,cv2.LINE_AA)
            elif (emotion == 'HAPPY'):
                cv2.putText(img, emotion, (x, y + h), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: ' + '%.4f' % prediction_emo[0][3], (x + w, y + h), font, 0.75, (0, 255, 255), 2,
                            cv2.LINE_AA)
            elif (emotion == 'NEUTRAL'):
                cv2.putText(img, emotion, (x, y + h), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: ' + '%.4f' % prediction_emo[0][4], (x + w, y + h), font, 0.75, (0, 255, 255), 2,
                            cv2.LINE_AA)
            elif (emotion == 'SAD'):
                cv2.putText(img, emotion, (x, y + h), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: ' + '%.4f' % prediction_emo[0][5], (x + w, y + h), font, 0.75, (0, 255, 255), 2,
                            cv2.LINE_AA)
            elif (emotion == 'SURPRISE'):
                cv2.putText(img, emotion, (x, y + h), font, 1, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'CONF: ' + '%.4f' % prediction_emo[0][6], (x + w, y + h), font, 0.75, (0, 255, 255), 2,
                            cv2.LINE_AA)
        cv2.imshow('img',img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

getFrame()