import cv2
import numpy as np
import torch
# from yolov5.utils import plots
import random
import time
from .write_logs import write_logs, time_synchronized
from datetime import datetime
from .utils import thermal_detect, plot_one_box

save_txt = 1
width = 1920
height = 1080

thermal_coords = {'x0': 0, 'y0': 0, 'x1': 640, 'y1': 480}

visual_coords = {'x0': thermal_coords['x1'],
                 'y0': 0,
                 'x1': 1920,
                 'y1': 720}

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='yolov5m_best.pt')  # custom model

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

webcam = cv2.VideoCapture(0)

if save_txt:
    filename = datetime.today().strftime('%Y-%m-%d')
    with open(filename + '.txt', 'a') as f:
        f.write('detection starts at ' + filename + '\n')

while True:
    ret, frame = webcam.read()
    # (height, width) = frame.shape[:2]
    frame = cv2.resize(frame, (width, height))  # for testing on webcam
    cv2.imshow('OG', frame)

    # Thermal Stream
    thermal = frame[thermal_coords['y0']:thermal_coords['y1'], thermal_coords['x0']:thermal_coords['x1']]
    thermal_detect(thermal)
    cv2.imshow('Thermal', thermal)
    thermal_logs = ' '
    # Thermal Detections

    # Visual Stream
    visual = frame[visual_coords['y0']:visual_coords['y1'], visual_coords['x0']:visual_coords['x1']]
    cv2.imshow('Visual', visual)

    # Visual Detections
    timestamp = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    t1 = time_synchronized()
    visual_results = model(visual)
    t2 = time_synchronized()
    visual_processed_time = t2 - t1

    detection_results = np.array(visual_results.xyxy[0])

    for i, info in enumerate(detection_results):  # detections per image
        # Write results
        xyxy = [info[0], info[1], info[2], info[3]]
        conf = info[4]
        cls = info[5]
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, visual, label=label, color=colors[int(cls)], line_thickness=3)
        visual_logs = timestamp + f' ,#{i+1} {label}'
        cv2.imshow('Visual', visual)
        cv2.waitKey(1)  # 1 millisecond

    # write logs
    if save_txt:
        logs = thermal_logs + visual_logs
        write_logs(filename, logs)

    # Concatenate processed videos to one screen
    horizontal_concat = np.zeros((height, width, 3), np.uint8)  # create an empty numpy array with full dimension in rgb
    horizontal_concat[:thermal_coords['y1'], :thermal_coords['x1'], :3] = thermal
    horizontal_concat[:visual_coords['y1'], visual_coords['x0']:visual_coords['x1'], :3] = visual
    cv2.putText(horizontal_concat, 'TEST INFO',
                (round(visual_coords['x0'] / 2), round(thermal_coords['y1'] + height / 2)), 0, 3, (255, 255, 255), 0,
                cv2.LINE_AA)  # display useful info
    cv2.imshow('Processed Video', horizontal_concat)

    if cv2.waitKey(1) == 27:
        exit(0)
