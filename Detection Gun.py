import numpy as np
import torch
import pandas as pd
import cv2
import time
from time import sleep
from os import system, name


def clear():
  
    # for windows
    if name == 'nt':
        _ = system('cls')
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def inference(img, model):

    # Inference
    
    results = model(img)

    # Results
    
    data_detection = results.pandas().xyxy[0]

    text = 'Number of guns: ' + str(data_detection.shape[0])

    for i in range(data_detection.shape[0]):
         yminn = int(data_detection['ymin'][i])
         ymaxx = int(data_detection['ymax'][i])
         xminn = int(data_detection['xmin'][i])
         xmaxx = int(data_detection['xmax'][i])

         cv2.rectangle(img, (xmaxx,ymaxx), (xminn, yminn), (0,0,255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (50,50), font, 1, (0,0,255), 2, cv2.LINE_AA)

    inference_time = time.time()
    clear()
    print("Running Inference, just a moment")
    print("")
    print("Time Elapsed: ", inference_time - start)
    

    return img


if __name__ == "__main__":

    start = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt')
    #model.classes = [0]

    video = cv2.VideoCapture('yt1s.com - Desfile dia da Infantaria Exército Brasileiro 20 BIB_Trim.mp4')

    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('DetectedG.avi', fourcc, 30, (width, height), isColor=True)

    while video.isOpened():
        success, frame = video.read()
        if success == False:
            break
        frames = inference(frame, model)
        frames = cv2.resize(frames, (width, height))
        out.write(frames)

    end = time.time()
    clear()
    print("Inference finished, video awaiting on this directory, ty for waiting")
    print("")
    print ("Total time: ", end - start)
    sleep(2)

    video.release()
    out.release()



