import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
import requests

tf.logging.set_verbosity(tf.logging.ERROR)

def pred_gender(roi):

    roi=cv2.resize(roi,(128,128))
    roi=np.reshape(roi,(1,128,128,1))
    res=model_g.predict(roi)
    return res

def pred_emotion(roi, model_e):

    roi=cv2.resize(roi,(64,64))
    roi=np.reshape(roi,(1,64,64,1))
    res=model_e.predict(roi)
    return res

#
# If cap_source is 0, use default camera
# If cap_source is 1, use in_file_name
#

def detectFacesVideo(cap_source, wv, out_file_name, in_file_name, show, model_number, model_e):
    
    if cap_source == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(in_file_name)

    if not cap.isOpened():
        print()
        print("OpenCV Warnings Currently Cannot Be Suppressed in Python\n")
        raise Exception()
        

    stats_emo_0 = [["Angry", 0],["Disgusted", 0], ["Fearful", 0], ["Happy", 0], ["Neutral", 0], ["Sad", 0], ["Surprised", 0]]
    stats_emo_1 = [["Angry", 0], ["Fearful", 0], ["Happy", 0], ["Neutral", 0], ["Sad", 0], ["Surprised", 0]]
    stats_emo = [stats_emo_0, stats_emo_1]

    stats_gen = [["Female", 0], ["Male", 0]]

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
                    stats_emo = None
                    stats_gen = None
                    raise
            else:
                vid_writer = cv2.VideoWriter(out_file_name, fourcc, 25, (frame_width, frame_height))
        else:
            print("No output video name specified!")
            return

    while(cap.isOpened()):
        
        ret,frame = cap.read()
        
        if not ret:
            break

        img = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img,1.2,5)

        #If this is the first time we've seen multiple faces, set multiple_faces to true and set stats to None
        if not multiple_faces:
            if len(faces) > 1:
                multiple_faces = True
                stats_emo = None
                stats_gen = None

        #Only increment number of frames if a face was found
        if len(faces) > 0:
            frames += 1

        for (x, y, w, h) in faces:

            scaled_prediction_size = (2*h)/356
            scaled_conf_size = (1*h)/356
            scaled_pixel_offset = int((25*h)/356)
            scaled_font_width = max(int((2*h)/356), 1)

            roi = frame[y:y + h, x:x + w]
            
            prediction_gen = pred_gender(roi/255.0)
            prediction_emo = pred_emotion(roi/255.0, model_e)
            
            pos_g=np.argmax(prediction_gen)
            pos_e=np.argmax(prediction_emo)
            pos_g=int(pos_g)
            pos_e=int(pos_e)
            #If there's only one face in the frame, 
            #Find the emotion and increment the number of frames that that emotion has been recognized
            if not multiple_faces:
                stats_emo[model_number][pos_e][1] += 1
                stats_gen[pos_g][1] += 1


            gender = GENDERS[pos_g]
            g_color = GENDER_COLORS[pos_g]

            cv2.rectangle(img, (x, y), (x + w, y + h), g_color, 1)

            cv2.putText(img, gender, (x, y), font, scaled_prediction_size, g_color, scaled_font_width, cv2.LINE_AA)
            cv2.putText(img, 'Conf: '+'%.3f'%prediction_gen[0][pos_g],(x,y+scaled_pixel_offset), font, scaled_conf_size, (0, 255, 255), scaled_font_width, cv2.LINE_AA)

            emotion = EMOTIONS[model_number][pos_e]
            e_color = EMOTION_COLORS[model_number][pos_e]

            cv2.putText(img, emotion, (x, y+h), font, scaled_prediction_size, e_color, scaled_font_width, cv2.LINE_AA)
            cv2.putText(img, 'Conf: '+'%.3f'%prediction_emo[0][pos_e],(x,y+h+scaled_pixel_offset), font, scaled_conf_size, (0,255,255),scaled_font_width,cv2.LINE_AA)

        k = cv2.waitKey(1) & 0xff

        #Write video
        if wv:
            vid_writer.write(img)
        
        if show:
            cv2.imshow('', img)
        
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    #Convert stats to percentage of frames
    if not stats_emo == None and not stats_gen == None:
        if frames != 0:
            for emotion in stats_emo[model_number]:
                emotion[1] = emotion[1]/frames * 100
        stats_video = [stats_emo[model_number], stats_gen]
    else:
        stats_video = None
    
    return stats_video

def detectFacesImage(in_file_name, image, model_number, model_e):
    if not in_file_name:
        img = image
    else:
        try:
            img = cv2.imread(in_file_name, cv2.IMREAD_COLOR)
        except Exception as e:
            raise
    
    stats_general_emotion_0= [["Angry", 0],["Disgusted", 0], ["Fearful", 0], ["Happy", 0], ["Neutral", 0], ["Sad", 0], ["Surprised", 0]]
    stats_general_emotion_1=[["Angry", 0], ["Fearful", 0], ["Happy", 0], ["Neutral", 0], ["Sad", 0], ["Surprised", 0]]
    stats_general_emotion =[stats_general_emotion_0, stats_general_emotion_1]
    stats_general_gender = [["Women", 0],["Men", 0]]
    stats_specific_emotion = []
    stats_specific_gender = []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.2,5)
    face_number = 1

    for (x, y, w, h) in faces:
        
        scaled_prediction_size = (2*h)/356
        scaled_conf_size = (1*h)/356
        scaled_pixel_offset = int((25*h)/356)
        scaled_font_width = max(int((2*h)/356), 1)

        roi = gray[y:y + h, x:x + w]
            
        prediction_gen = pred_gender(roi/255.0)
        prediction_emo = pred_emotion(roi/255.0, model_e)
            
        pos_g=np.argmax(prediction_gen)
        pos_e=np.argmax(prediction_emo)

        gender = GENDERS[pos_g]
        g_color = GENDER_COLORS[pos_g]

        emotion = EMOTIONS[model_number][pos_e]
        e_color = EMOTION_COLORS[model_number][pos_e]

        stats_general_emotion[model_number][pos_e][1] += 1
        stats_general_gender[pos_g][1] += 1

        stats_specific_emotion.append([emotion,prediction_emo[0][pos_e]])
        stats_specific_gender.append([gender,prediction_gen[0][pos_g]])

        cv2.rectangle(img, (x, y), (x + w, y + h), g_color, 1)

        cv2.putText(img, gender, (x, y), font, scaled_prediction_size, g_color, scaled_font_width, cv2.LINE_AA)
        cv2.putText(img, 'Conf: '+'%.3f'%prediction_gen[0][pos_g],(x,y+scaled_pixel_offset), font, scaled_conf_size, (0, 255, 255), scaled_font_width, cv2.LINE_AA)

        cv2.putText(img, emotion, (x, y+h), font, scaled_prediction_size, e_color, scaled_font_width, cv2.LINE_AA)
        cv2.putText(img, 'Conf: '+'%.3f'%prediction_emo[0][pos_e],(x,y+h+scaled_pixel_offset), font, scaled_conf_size, (0,255,255),scaled_font_width,cv2.LINE_AA)

        #Label each face to match to stats later
        cv2.putText(img, str(face_number), (x+w, y), font, scaled_prediction_size, (0, 255, 255), scaled_font_width, cv2.LINE_AA)

        face_number += 1
    
    cv2.imshow('', img)
    k = cv2.waitKey(30) & 0xff
    
    write_image = input("Would you like to save the image with detected faces? (y/n)")
    
    print()
    
    if write_image == 'y':
        out_file_name = "OutputImages/" + input("Name your output image (include .png extension): ")
        print()
        cv2.imwrite(out_file_name, img)
    
    
    cv2.destroyAllWindows()
    stats_image = [stats_general_emotion[model_number], stats_general_gender, stats_specific_emotion, stats_specific_gender]
   
    if len(faces) == 0:
        stats_image = None
        
    return stats_image

def takePhoto(model_number, model_e):
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception as e:
        raise

    if not cap.isOpened():
        raise Exception()
    
    frame_final = None

    while(True):
        ret,frame = cap.read()
        
        k = cv2.waitKey(30) & 0xff
        
        cv2.namedWindow('')
        
        cv2.imshow('', frame)

        if k == ord('p'):
            while True:
                k2 = cv2.waitKey(30) & 0xff
                if k2 == ord('p'):
                    break
                if k2 == ord('s'):
                    stats_image = detectFacesImage(None, frame, model_number, model_number, model_e)
                    return stats_image
                if k2 == 27:
                    return None
        
        if k == 27:
            break
    return None
    

#---------------------------CLASSIFIER ARRAYS AND MODELS-----------------------------#
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_g=load_model('saved_weights_gender/weights-17-0.134326-0.950.hdf5')

#----------------------------MODEL AND COLOR CONSTANTS--------------------------------#
EMOTIONS_1 = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
EMOTIONS_2 = ["Angry", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
EMOTIONS = [EMOTIONS_1, EMOTIONS_2]

EMOTION_COLORS_1= [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 165, 255)]
EMOTION_COLORS_2= [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 165, 255)]
EMOTION_COLORS = [EMOTION_COLORS_1, EMOTION_COLORS_2]

GENDERS = ["Woman", "Man"]
GENDER_COLORS = [(0, 0, 255), (255, 0, 0)]
    
font = cv2.FONT_HERSHEY_SIMPLEX

def chooseModel(model_choice):
    if model_choice == '1':
        model_exp=load_model('saved_weights_emotion/weights-33-1.103463-0.601.hdf5')
        model_num = 0
    elif model_choice == '2':
        model_exp=load_model('saved_weights_emotion/weights-73-1.031777-0.613.hdf5')
        model_num = 1
    elif model_choice == '3':
        model_exp=load_model('saved_weights_emotion/weights-83-1.146881-0.572.hdf5')
        model_num = 1
    else:
        model_exp=load_model('saved_weights_emotion/weights-33-1.103463-0.601.hdf5')
        model_num = 0
    return model_exp, model_num

def main(debugging):

    print("--------------------------------")
    print("------------Welcome-------------")
    print("--------------------------------\n")
    print("Select emotion model to run:")
    print("Model (1): Full Dataset")
    print("Model (2): Partial Dataset without Disgust")
    print("Model (3): Normalized Dataset wihtout Disgust")

    response = input("Choose Expression Recognition model:")
    print()
    model_e, model = chooseModel(response)



    while(True):
        print()
        print("Choose an option:")
        print("0: Remote device (Smartphone)")
        print("1: Video (Default Device Camera)")
        print("2: Video (Pre Recorded)")
        print("3: Take Photo")
        print("4: Photo (Pre Taken)")
        print("5: Change model")
        print("6: Exit\n")
        

        choice = input("Your choice: ")
        print()
        wv = False
        wi = False
        out_file_name = None
        in_file_name = None
        stats_video = None
        stats_image = None


        if choice=='0':
            url = 'http://10.26.195.82:8080/shot.jpg'

            while True:
                img_resp = requests.get(url)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(img_arr, -1)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(img, 1.2, 5)
                face_number = 1

                for (x, y, w, h) in faces:
                    scaled_prediction_size = (2 * h) / 356
                    scaled_conf_size = (1 * h) / 356
                    scaled_pixel_offset = int((25 * h) / 356)
                    scaled_font_width = max(int((2 * h) / 356), 1)

                    roi = gray[y:y + h, x:x + w]

                    prediction_gen = pred_gender(roi / 255.0)
                    prediction_emo = pred_emotion(roi / 255.0, model_e)

                    pos_g = np.argmax(prediction_gen)
                    pos_e = np.argmax(prediction_emo)

                    gender = GENDERS[pos_g]
                    g_color = GENDER_COLORS[pos_g]

                    emotion = EMOTIONS[model][pos_e]
                    e_color = EMOTION_COLORS[model][pos_e]

                    # stats_general_emotion[model][pos_e][1] += 1
                    # stats_general_gender[pos_g][1] += 1
                    #
                    # stats_specific_emotion.append([emotion, prediction_emo[0][pos_e]])
                    # stats_specific_gender.append([gender, prediction_gen[0][pos_g]])

                    cv2.rectangle(img, (x, y), (x + w, y + h), g_color, 1)

                    cv2.putText(img, gender, (x, y), font, scaled_prediction_size, g_color, scaled_font_width,
                                cv2.LINE_AA)
                    cv2.putText(img, 'Conf: ' + '%.3f' % prediction_gen[0][pos_g], (x, y + scaled_pixel_offset), font,
                                scaled_conf_size, (0, 255, 255), scaled_font_width, cv2.LINE_AA)

                    cv2.putText(img, emotion, (x, y + h), font, scaled_prediction_size, e_color, scaled_font_width,
                                cv2.LINE_AA)
                    cv2.putText(img, 'Conf: ' + '%.3f' % prediction_emo[0][pos_e], (x, y + h + scaled_pixel_offset),
                                font, scaled_conf_size, (0, 255, 255), scaled_font_width, cv2.LINE_AA)

                    # Label each face to match to stats later
                    cv2.putText(img, str(face_number), (x + w, y), font, scaled_prediction_size, (0, 255, 255),
                                scaled_font_width, cv2.LINE_AA)

                    face_number += 1



                cv2.imshow('androCam', img)

                if cv2.waitKey(1) == 27:
                    break
            cv2.destroyAllWindows()





        if choice == '1' or choice == '2' or choice == '3' or choice == '4' or choice == '5' or choice == '6':
            if choice == '6':
                break
            if choice == '5':
                print("Select emotion model to run:")
                print("Model (1): Full Dataset")
                print("Model (2): Partial Dataset without Disgust")
                print("Model (3): Normalized Dataset wihtout Disgust")
                print()

                response = input("Choose Expression Recognition model:")
                print()
                model_e, model = chooseModel(response)
                continue

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
                        stats_video = detectFacesVideo(0, wv, out_file_name, in_file_name, True, model, model_e)
                    except Exception as e:
                        print("Cannot read/write video!\n")
                        if debugging:
                            print(e)
                        continue
            
                #Input video from given source
                elif choice == '2':
                    in_file_name = "InputVideos/" + input("Enter path to input video location:")
                    print()

                    if write_video:
                        ask = input("Would you like to see the video as it is processed?(y/n)")
                        if ask == 'y':
                            print()
                            show = True
                        else:
                            show = False
                    else:
                        show = True
                    try:
                        print()
                        stats_video = detectFacesVideo(1, wv, out_file_name, in_file_name, show, model, model_e)

                    except Exception as e:
                        print("Cannot find specified video!\n")
                        if debugging:
                            print(e)
                        continue


            #Take photo
            elif choice == '3':
                print()
                print("Press p to pause or resume video, press s to save paused frame, press esc to exit")
                try:
                    stats_image = takePhoto(model, model_e)
                except Exception as e:
                    print("Cannot open camera!")
                    if debugging:
                        print(e)
                    continue
                
            #Input image
            else:
                in_file_name = "InputImages/" + input("Enter input photo name:")
                print()
                try:
                    stats_image = detectFacesImage(in_file_name, None, model, model_e)
                except Exception as e:
                    print("Cannot process input photo!")
                    if debugging:
                        print(e)
                    continue

        else:
            print("Select one of the options\n")
            continue

        if choice == '1' or choice == '2':
            stat_ask = input("Would you like to view prediction stats for the previous video? (y/n)")
            print()
        
            if stat_ask == 'y':
                if not stats_video == None:
                    print('----------PREDICTION STATS---------')
                    print("Time feeling each emotion:\n")
                    for emotion in stats_video[0]:
                        print(emotion[0]+' '+str(np.around(emotion[1],4))+'%')
                    print()
                    print("Predicted gender:")
                    g_frames = []
                    for gender in stats_video[1]:
                        g_frames.append(gender[1])
                    g = np.argmax(g_frames)
                    print(stats_video[1][g][0])
                    print('-----------------------------------')
                else:
                    print("Sorry, multiple faces were detected. Prediction tats for multiple faces in video are currently unsupported.\n")
                    continue
        
        if choice == '3' or choice == '4':
            stat_ask = input("Would you like to view prediction stats for the image? (y/n)")
            print()
            if stat_ask == 'y':
                if stats_image:
                    print('----------PREDICTION STATS----------')
                    print("Number of people feeling each emotion:")
                    for emotion in stats_image[0]:
                        if emotion[1] == 1:
                            print(emotion[0]+': '+str(emotion[1]) + ' person')
                        else:        
                            print(emotion[0]+': '+str(emotion[1]) + ' people')
                    print()
                    for gender in stats_image[1]:
                        print(gender[0]+': '+str(gender[1]))
                    print()
                    print("Specific predictions for each detected person (see saved image for who's who):")
                    print()
                    for specific in range(len(stats_image[2])):
                        print("-----------------")
                        print("Person " + str(specific + 1) + ':')
                        print("Predicted emotion: " + stats_image[2][specific][0])
                        print("Confidence: " + str(stats_image[2][specific][1]))
                        print("Predicted gender: " +  stats_image[3][specific][0])
                        print("Confidence: " + str(stats_image[3][specific][1]))
                        print("-----------------\n")
                    print('-----------------------------------\n')
                else:
                    print('No faces detected in the image!')
                    continue
                
        write_file = input('Write stats to file?(y/n)')
        if write_file == 'y':
            text_file_name = "OutputStats/" + input('Name your text file (with .txt extension):')
            print()
            fout = open(text_file_name, "w+")
            if choice == '1' or choice == '2':
                fout.write("Predicted prevalence of each emotion:\n")
                for emotion in stats_video[0]:
                    fout.write(emotion[0]+': '+str(np.around(emotion[1],4))+'%'+'\n')
                g_pos = np.argmax(stats_video[1][1])
                fout.write("Predicted gender: " + str(stats_video[1][g_pos][1]))
            else:
                fout.write("Predicted number of people feeling each emotion:\n")
                for emotion in stats_image[0]:
                    fout.write(emotion[0] + ': ' + str(emotion[1]) + '\n')
                fout.write('\n')
                fout.write("Predicted number of men and women:\n")
                for gender in stats_image[1]:
                    fout.write(gender[0] + ': ' + str(gender[1]) + '\n')
                fout.write('\n')
                fout.write("Specific predictions for faces (see image for who's who):\n")
                for specific in range(len(stats_image[2])):
                        fout.write("Person " + str(specific + 1) + ':\n')
                        fout.write("Predicted emotion: " + stats_image[2][specific][0] + '\n')
                        fout.write("Confidence: " + str(stats_image[2][specific][1]) + '\n')
                        fout.write("Predicted gender: " +  stats_image[3][specific][0] + '\n')
                        fout.write("Confidence: " + str(stats_image[3][specific][1]) + '\n')
                        fout.write('\n')
            fout.close()

main(True)