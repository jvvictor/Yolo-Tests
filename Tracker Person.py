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

def inference(img, model, objectsX, objectsY, framecount):

    # Inference
    
    results = model(img)

    # Results
    
    list_y = []
    list_x = []

    list_y = objectsY
    list_x = objectsX

    font = cv2.FONT_HERSHEY_SIMPLEX

    data_detection = results.pandas().xyxy[0]

    text = 'Number of persons: ' + str(data_detection.shape[0])
    
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
            
   
    cv2.putText(img, text, (50,50), font, 1, (0,0,255), 2, cv2.LINE_AA)


    inference_time = time.time()
    clear()
    print("Running Inference, just a moment")
    print("")
    print("Time Elapsed: ", inference_time - start)
    

    return img, list_x, list_y, count


if __name__ == "__main__":

    start = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.classes = [0]

    video = cv2.VideoCapture('test.mp4')

    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('TrackedP.avi', fourcc, 30, (width, height), isColor=True)

    objectsX = []
    objectsY = []
    framecount = 0

    while video.isOpened():
        framecount = framecount + 1
        success, frame = video.read()
        if success == False:
            break
        frames, objectsX, objectsY, wtf = inference(frame, model, objectsX, objectsY, framecount)
        frames = cv2.resize(frames, (width, height))
        out.write(frames)

    end = time.time()
    clear()
    print("Inference finished, video awaiting in this directory, ty for waiting")
    print("")
    print ("Total time: ", end - start)
    sleep(2)

    video.release()
    out.release()