import numpy as np
import torch
import pandas as pd
import cv2
import time
from time import sleep
from os import system, name
import math


def clear():
  
    # for windows
    if name == 'nt':
        _ = system('cls')
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def inference(img, model, objectsX, objectsY, framecount, search, type):

    # Inference
    
    results = model(img)

    # Results
    
    list_y = []
    list_x = []

    list_y = objectsY
    list_x = objectsX

    font = cv2.FONT_HERSHEY_SIMPLEX

    data_detection = results.pandas().xyxy[0]

    if search == 1:
        text = 'Number of persons: ' + str(data_detection.shape[0])
    else:
        text = 'Number of guns: ' + str(data_detection.shape[0])
    
    count = 0

    for i in range(data_detection.shape[0]):

        track = 0
        yminn = int(data_detection['ymin'][i])
        ymaxx = int(data_detection['ymax'][i])
        xminn = int(data_detection['xmin'][i])
        xmaxx = int(data_detection['xmax'][i])

        cv2.rectangle(img, (xmaxx,ymaxx), (xminn, yminn), (0,0,255), 2)

        center_ret_X = (xminn + (xmaxx/2))
        center_ret_Y = (yminn + (ymaxx/2))

        if type == 1:
            if framecount == 1:
                if i == 0:
                    list_y.append(center_ret_Y)
                    list_x.append(center_ret_X)
                    id = 'ID: ' + str(1)
                    track = 1
                    cv2.putText(img, id, (xmaxx,ymaxx), font, 1, (0,0,255), 2, cv2.LINE_AA)
                    framecount = 2
            else:
                for j in range(len(list_y)):
                    d = (((center_ret_X - list_x[j])**2) + ((center_ret_Y - list_y[j])**2))**0.5
                    if (d < 87) and (track == 0):
                        track = 1
                        minor = d
                        position = j
                    if (d < 87) and (track == 1) and (d < minor):
                        minor = d 
                        position = j
            
                if track == 1:
                    list_x[position] = center_ret_X
                    list_y[position] = center_ret_Y
                    id = 'ID: ' + str(position)
                    cv2.putText(img, id, (xmaxx,ymaxx), font, 1, (0,0,255), 2, cv2.LINE_AA)
            
                if track == 0: 
                    list_y.append(center_ret_Y)
                    list_x.append(center_ret_X)
                    id = 'ID: ' + str(len(objectsY))
                    cv2.putText(img, id, (xmaxx,ymaxx), font, 1, (0,0,255), 2, cv2.LINE_AA)
            
   
    cv2.putText(img, text, (50,50), font, 1, (255,0,0), 2, cv2.LINE_AA)


    inference_time = time.time()
    clear()
    print("Running Inference, just a moment")
    print("")
    print("Time Elapsed: ", inference_time - start)
    

    return img, list_x, list_y, count


if __name__ == "__main__":

    clear()

    video_name = str(input("Write the name of the video, or the relative path: "))

    clear()

    search = int(input("What you want to search? (1 for Person - 2 for Guns): "))

    clear()

    while (search != 1) and (search != 2):
        print("Invalid Input")
        print("")
        search = int(input("What you want to search? (1 for Person - 2 for Guns): "))

    clear()

    type = int(input("Want to track? (1 for Yes - 2 for No): "))

    clear()

    while (type != 1) and (type != 2):
        print("Invalid Input")
        print("")
        type = int(input("Want to track? (1 for Yes - 2 for No): "))

    clear()

    if search == 1:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model.classes = [0]
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best50.pt')

    start = time.time()

    video = cv2.VideoCapture(filename = video_name)

    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    if search == 1:
        if type == 1:
            out = cv2.VideoWriter('TrackedPerson.avi', fourcc, 30, (width, height), isColor=True)
            name = "TrackedPerson.avi"
        else:
            out = cv2.VideoWriter('DetectedPerson.avi', fourcc, 30, (width, height), isColor=True)
            name = "DetectedPerson.avi"
    if search == 2:
        if type == 1:
            out = cv2.VideoWriter('TrackedGun.avi', fourcc, 30, (width, height), isColor=True)
            name = "TrackedGun.avi"
        else:
            out = cv2.VideoWriter('DetectedGun.avi', fourcc, 30, (width, height), isColor=True)
            name = "DetectedGun.avi"


    objectsX = []
    objectsY = []
    framecount = 0

    while video.isOpened():
        framecount = framecount + 1
        success, frame = video.read()
        if success == False:
            break
        frames, objectsX, objectsY, wtf = inference(frame, model, objectsX, objectsY, framecount, search, type)
        frames = cv2.resize(frames, (width, height))
        out.write(frames)

    end = time.time()
    clear()
    print("Inference finished, video awaiting in this directory as ", name ,", ty for waiting")
    print("")
    print ("Total time: ", end - start)
    sleep(2)

    video.release()
    out.release()