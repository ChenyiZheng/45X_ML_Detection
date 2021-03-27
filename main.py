import cv2
import numpy as np
import torch
import random
from datetime import datetime
from working_scripts.utils import thermal_detect, plot_one_box, write_logs, time_synchronized, crop_image

save_txt = 1
filename = datetime.today().strftime('%Y-%m-%dT%H%M%S%z')
# width = 1920
# height = 1080

thermal_aspect = {'width': 4, 'height': 3}
visual_aspect = {'width': 4, 'height': 3}

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='yolov5m_best.pt')  # custom model

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# webcam = cv2.VideoCapture(0)

webcam = cv2.VideoCapture('ThermVisVid4x3.mp4')

while True:
    ret, frame = webcam.read()
    # cv2.imshow('OG', frame)

    thermal, visual = crop_image(frame, thermal_aspect, visual_aspect)
    # Thermal Stream
    thermal, tot_area, num_hotspots = thermal_detect(thermal)
    cv2.imshow('Thermal', thermal)

    # Visual Stream
    # cv2.imshow('Visual', visual)

    # Visual Detections
    timestamp = datetime.today().strftime('%H:%M:%S')
    t1 = time_synchronized()
    visual_results = model(visual)
    t2 = time_synchronized()
    visual_processed_time = round(t2 - t1, 3)

    detection_results = np.array(visual_results.xyxy[0])

    visual_logs = ' '
    for i, info in enumerate(detection_results):  # detections per image
        if detection_results.size:
            # Write results
            xyxy = [info[0], info[1], info[2], info[3]]
            conf = info[4]
            cls = info[5]
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, visual, label=label, color=colors[int(cls)], line_thickness=3)
            visual_logs = visual_logs + f'#{i + 1} {label}, '
            cv2.imshow('Visual', visual)
            cv2.waitKey(1)  # 1 millisecond
        else:
            visual_logs = 'None '

    # write logs
    if save_txt:
        write_logs(filename, num_hotspots, tot_area, visual_logs, timestamp, visual_processed_time)

    # # Concatenate processed videos to one screen
    # horizontal_concat = np.zeros((height, width, 3), np.uint8)  # create an empty numpy array with full dimension in rgb
    # horizontal_concat[:thermal_coords['y1'], :thermal_coords['x1'], :3] = thermal
    # horizontal_concat[:visual_coords['y1'], visual_coords['x0']:visual_coords['x1'], :3] = visual
    # cv2.putText(horizontal_concat, 'TEST INFO',
    #             (round(visual_coords['x0'] / 2), round(thermal_coords['y1'] + height / 2)), 0, 3, (255, 255, 255), 0,
    #             cv2.LINE_AA)  # display useful info
    # cv2.imshow('Processed Video', horizontal_concat)

    if cv2.waitKey(1) == 27:
        exit(0)
