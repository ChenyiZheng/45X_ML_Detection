import cv2
import numpy as np
import torch

thermal_coords = {'topleftx': 0, 'toplefty': 0, 'botrightx': 1920, 'botrighty': 1080}

visual_coords = {'topleftx': thermal_coords['topleftx'] + thermal_coords['toplefty'],
                 'toplefty': 0,
                 'botrightx': thermal_coords['botrightx'] + 640,
                 'botrighty': 360}

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='yolov5x_best.pt')  # custom model

webcam = cv2.VideoCapture(1)

# (x, y, w, h) = cv2.boundingRect(c)
# cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 20)
# roi = frame[y:y+h, x:x+w]

while True:
    ret, frame = webcam.read()
    # (height, width) = frame.shape[:2]
    thermal = frame[thermal_coords['toplefty']:thermal_coords['botrighty'], thermal_coords['topleftx']:thermal_coords['botrightx']]

    cv2.imshow('Video', thermal)

    visual = frame[visual_coords['toplefty']:visual_coords['botrighty'], visual_coords['topleftx']:visual_coords['botrightx']]

    visual_results = model(visual)
    visual_results = np.array(visual_results.xyxy[0])
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