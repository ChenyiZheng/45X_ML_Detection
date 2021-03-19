import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='yolov5s_voc_best.pt')  # custom model

cap = cv2.VideoCapture('http://89.29.108.38:80/mjpg/video.mjpg')

# (x, y, w, h) = cv2.boundingRect(c)
# cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 20)
# roi = frame[y:y+h, x:x+w]

while True:
    ret, frame = cap.read()
    # (height, width) = frame.shape[:2]
    sky = frame[0:100, 0:200]
    cv2.imshow('Video', sky)

    if cv2.waitKey(1) == 27:
        exit(0)