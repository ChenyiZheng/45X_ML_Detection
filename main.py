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

# (x, y, w, h) = cv2.boundingRect(c)
# cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 20)
# roi = frame[y:y+h, x:x+w]

if save_txt:
    filename = datetime.today().strftime('%Y-%m-%d')
    with open(filename + 'txt', 'a') as f:
        f.write('detection starts at ' + '.txt' + '\n')

while True:
    ret, frame = webcam.read()
    # (height, width) = frame.shape[:2]
    frame = cv2.resize(frame, (width, height))  # for testing on webcam
    cv2.imshow('OG', frame)

    # Thermal Stream
    thermal = frame[thermal_coords['y0']:thermal_coords['y1'], thermal_coords['x0']:thermal_coords['x1']]
    thermal_detect(thermal, [0, 0, 0])
    cv2.imshow('Thermal', thermal)

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
        log = f'{timestamp} {label} {i} Done. ({t2 - t1:.3f}s)'
        write_logs(filename, log)

        cv2.imshow('Visual', visual)
        cv2.waitKey(1)  # 1 millisecond

    # structure of array
    #                   x1           y1           x2           y2   confidence        class
    # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
    #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
    #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])

    # (x, y, w, h) = cv2.boundingRect(c)
    # cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 20)
    # roi = frame[y:y+h, x:x+w]

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
