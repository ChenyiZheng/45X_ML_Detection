import cv2
import numpy as np
import torch
# from yolov5.utils import plots
import random


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


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

while True:
    ret, frame = webcam.read()
    # (height, width) = frame.shape[:2]
    cv2.imshow('OG', frame)
    thermal = frame[thermal_coords['y0']:thermal_coords['y1'], thermal_coords['x0']:thermal_coords['x1']]

    cv2.imshow('Thermal', thermal)

    visual = frame[visual_coords['y0']:visual_coords['y1'], visual_coords['x0']:visual_coords['x1']]

    cv2.imshow('Visual', visual)

    visual_results = model(visual)
    visual_results = np.array(visual_results.xyxy[0])
    # Process detections
    for i, det in enumerate(visual_results):  # detections per image
        # Write results
        xyxy = [det[0], det[1], det[2], det[3]]
        conf = det[4]
        cls = det[5]
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, visual, label=label, color=colors[int(cls)], line_thickness=3)

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

    if cv2.waitKey(1) == 27:
        exit(0)