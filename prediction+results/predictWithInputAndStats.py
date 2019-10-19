import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

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

#
# If cap_source is 0, use default camera
# If cap_source is 1, use in_file_name
#

def detectFacesVideo(cap_source, wv, out_file_name, in_file_name, show):
    try:
        if cap_source == 0:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(in_file_name)
    except cv2.error as e:
        raise
    
    if not cap.isOpened():
        raise Exception()

    stats = [["Angry", 0],["Disgusted", 0], ["Fearful", 0], ["Happy", 0], ["Neutral", 0], ["Sad", 0], ["Surprised", 0]]
    multiple_faces = False
    frames = 0
    
    #If writing a video, initialize the videowriter
    if wv:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        if out_file_name:
            if in_file_name:
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    vid_writer = cv2.VideoWriter(out_file_name, fourcc, fps, (frame_width, frame_height) )
                except Exception as e:
                    stats = None
                    raise
            else:
                vid_writer = cv2.VideoWriter(out_file_name, fourcc, 20, (frame_width, frame_height))
        else:
            print("No output video name specified!")
            return

    while(cap.isOpened()):
        
        ret,frame = cap.read()
        
        if not ret:
            break

        frames += 1

        img = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img,1.2,5)

        #If this is the first time we've seen multiple faces, set multiple_faces to true and set stats to None
        if not multiple_faces:
            if len(faces) > 1:
                multiple_faces = True
                stats = None

        for (x, y, w, h) in faces:

            scaled_prediction_size = (1.5*h)/356
            scaled_conf_size = (0.9*h)/356
            scaled_pixel_offset = int((25*h)/356)
            scaled_font_width = max(int((2*h)/356), 1)

            roi = frame[y:y + h, x:x + w]
            
            prediction_gen = pred_gender(roi/255.0)
            prediction_emo = pred_emotion(roi/255.0)
            
            pos_g=np.argmax(prediction_gen)
            pos_e=np.argmax(prediction_emo)

            #If there's only one face in the frame, 
            #Find the emotion and increment the number of frames that that emotion has been recognized
            if not multiple_faces:
                stats[pos_e][1] += 1

            gender = GENDERS[pos_g]
            g_color = GENDER_COLORS[pos_g]
            cv2.rectangle(img, (x, y), (x + w, y + h), g_color, 1)

            cv2.putText(img, gender, (x, y), font, scaled_prediction_size, g_color, scaled_font_width, cv2.LINE_AA)
            cv2.putText(img, 'Conf: '+'%.3f'%prediction_gen[0][pos_g],(x,y+scaled_pixel_offset), font, scaled_conf_size, (0, 255, 255), scaled_font_width, cv2.LINE_AA)

            emotion = EMOTIONS[pos_e]
            e_color = EMOTION_COLORS[pos_e]

            cv2.putText(img, emotion, (x, y+h), font, scaled_prediction_size, e_color, scaled_font_width, cv2.LINE_AA)
            cv2.putText(img, 'Conf: '+'%.3f'%prediction_emo[0][pos_e],(x,y+h+scaled_pixel_offset), font, scaled_conf_size, (0,255,255),scaled_font_width,cv2.LINE_AA)

        if wv:
            vid_writer.write(img)
        
        if show:
            cv2.imshow('', img)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    #Convert stats to percentage of frames
    if frames != 0 and stats:
        for emotion in stats:
            emotion[1] = emotion[1]/frames * 100
    
    return stats

#NOTE to self: figure out scaling factor to scale text with.
# Try x*10/w for images to see.

def detectFacesImage(in_file_name):
    try:
        img = cv2.imread(in_file_name, cv2.IMREAD_COLOR)
        frame = np.copy(img)
    except Exception as e:
        raise
    
    stats = [["Angry", 0],["Disgusted", 0], ["Fearful", 0], ["Happy", 0], ["Neutral", 0], ["Sad", 0], ["Surprised", 0]]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.2,5)

    for (x, y, w, h) in faces:
        
        scaled_prediction_size = (1.5*h)/356
        scaled_conf_size = (0.9*h)/356
        scaled_pixel_offset = int((25*h)/356)
        scaled_font_width = max(int((2*h)/356), 1)

        roi = gray[y:y + h, x:x + w]
            
        prediction_gen = pred_gender(roi/255.0)
        prediction_emo = pred_emotion(roi/255.0)
            
        pos_g=np.argmax(prediction_gen)
        pos_e=np.argmax(prediction_emo)
            
        stats[pos_e][1] += 1

        gender = GENDERS[pos_g]
        g_color = GENDER_COLORS[pos_g]

        cv2.rectangle(img, (x, y), (x + w, y + h), g_color, 1)

        cv2.putText(img, gender, (x, y), font, scaled_prediction_size, g_color, scaled_font_width, cv2.LINE_AA)
        cv2.putText(img, 'Conf: '+'%.3f'%prediction_gen[0][pos_g],(x,y+scaled_pixel_offset), font, scaled_conf_size, (0, 255, 255), scaled_font_width, cv2.LINE_AA)

        emotion = EMOTIONS[pos_e]
        e_color = EMOTION_COLORS[pos_e]

        cv2.putText(img, emotion, (x, y+h), font, scaled_prediction_size, e_color, scaled_font_width, cv2.LINE_AA)
        cv2.putText(img, 'Conf: '+'%.3f'%prediction_emo[0][pos_e],(x,y+h+scaled_pixel_offset), font, scaled_conf_size, (0,255,255),scaled_font_width,cv2.LINE_AA)
    
    if len(faces) == 0:
        stats = None
    
    cv2.imshow('', img)
    k = cv2.waitKey(30) & 0xff
    
    write_image = input("Would you like to save the image with detected faces? (y/n)")
    
    print()
    
    if write_image == 'y':
        out_file_name = "OutputImages/" + input("Name your output image (include .png extension): ")
        print()
        cv2.imwrite(out_file_name, img)
    
    cv2.destroyAllWindows()
    return stats




face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_g=load_model('saved_weights_gender/weights-17-0.134326-0.950.hdf5')
model_e=load_model('saved_weights_emotion/weights-33-1.103463-0.601.hdf5')

EMOTIONS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
EMOTION_COLORS = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 165, 255)]
GENDERS = ["Female", "Male"]
GENDER_COLORS = [(0, 0, 255), (255, 0, 0)]
    
font = cv2.FONT_HERSHEY_SIMPLEX


#
def main(debugging):
    
    while(True):
        print()
        print("Choose input source:")
        print("1: Default device camera")
        print("2: Pre-recorded video")
        print("3: Pre-taken photo")
        print("4: Exit\n")
        

        choice = input("Your choice: ")
        print()
        wv = False
        wi = False
        out_file_name = None
        in_file_name = None
        stats = None
    
        if choice == '1' or choice == '2' or choice == '3' or choice == '4':
            if choice == '4':
                break
            else:
                if choice == '1' or choice =='2':
                    write_video = input("Would you like to write video with detection? (y/n)")
                    print()
                    if write_video == 'y':
                        wv = True
                        out_file_name = "OutputVideos/" + input("Name your output file (include .avi extension): ")
                        print()
                    print("Press esc to end a video session.\n")

            #Input video from camera
            if choice == '1':
                try:
                    stats = detectFacesVideo(0, wv, out_file_name, in_file_name, True)
                except Exception as e:
                    print("Cannot read/write video!\n")
                    if debugging:
                        print(e)
                    continue
            
            #Input video from given source
            elif choice == '2':
                in_file_name = "InputVideos/" + input("Enter input video name:")
                print()

                if write_video:
                    ask = input("Would you like to see the video as it is processed?(y/n)")
                    if ask == 'y':
                        print()
                        show = True
                    else:
                        print()
                        print("Processing video...\n")
                        show = False
                else:
                    show = True
                try:
                    stats = detectFacesVideo(1, wv, out_file_name, in_file_name, show)
                
                except Exception as e:
                    print("Cannot find specified video!\n")
                    if debugging:
                        print(e)
                    continue
            
            #Input image
            else:
                in_file_name = "InputImages/" + input("Enter path to input photo:")
                print()
                try:
                    stats = detectFacesImage(in_file_name)
                except Exception as e:
                    print("Cannot process input photo!")
                    if debugging:
                        print(e)
                    continue

        else:
            print("Select one of the options\n")
            continue

        if choice == '1' or choice == '2':
            if not stats:
                print("Stats for this video are unavailable as multiple faces were detected.\nStats for multiple faces in video are currently unsupported.")
                continue
            else:
                stat_ask = input("Would you like to view emotion stats for the previous video? (y/n)")
                print()
        
            if stat_ask == 'y':
                if stats:
                    print('--------------STATS----------------')
                    print("Time feeling each emotion:\n")
                    for emotion in stats:
                        print(emotion[0]+' '+str(np.around(emotion[1],4))+'%')
                    print('-----------------------------------\n')
                else:
                    print("Sorry, multiple faces were detected. Stats for multiple faces in video are currently unsupported.\n")
                    continue
        
        if choice == '3':
            stat_ask = input("Would you like to view emotion stats for the image? (y/n)")
            print()
            if stat_ask == 'y':
                if stats:
                    print('--------------STATS----------------')
                    print("Number of people feeling each emotion:")
                    for emotion in stats:
                        if emotion[1] == 1:
                            print(emotion[0]+': '+str(emotion[1]) + ' person')
                        else:        
                            print(emotion[0]+': '+str(emotion[1]) + ' people')
                    print('-----------------------------------\n')
                else:
                    print('No faces detected in the image!')
                    continue
                
        write_file = input('Write stats to file?(y/n)')
        if write_file == 'y':
            text_file_name = "OutputStats/" + input('Name your text file (with .txt extension):')
            print()
            fout = open(text_file_name, "w+")
            for emotion in stats:
                if choice == '1' or choice == '2':
                    fout.write(emotion[0]+': '+str(np.around(emotion[1],4))+'%'+'\n')
                else:
                    if emotion[1] == 1:
                        fout.write(emotion[0]+': '+str(emotion[1]) + ' person\n')
                    else:        
                        fout.write(emotion[0]+': '+str(emotion[1]) + ' people\n')
            fout.close()
main(True)