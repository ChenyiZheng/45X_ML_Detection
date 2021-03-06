import cv2
import numpy as np
import torch
import random
import time
from datetime import datetime
from working_scripts.utils import thermal_detect, plot_one_box, write_logs, time_synchronized, crop_image, save_frames

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized
import argparse


def detect(weights: str,
           source: str,
           device: str,
           img_size: int,
           conf_thres: float,
           iou_thres: float,
           classes: int,
           agnostic_nms: str,
           augment: str,
           thermal_aspect=None,
           visual_aspect=None):

    device = select_device(device)  # 0 or 0,1,2,3 or cpu'
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    webcam = cv2.VideoCapture(source)

    while True:
        ret, frame = webcam.read()

        thermal = None
        visual = None

        if thermal_detect and visual_aspect:
            thermal, visual = crop_image(frame, thermal_aspect, visual_aspect)

        elif thermal_aspect:
            thermal = frame

        elif visual_aspect:
            visual = frame

        # Thermal Stream
        if thermal:
            thermal, tot_area, num_hotspots = thermal_detect(thermal)

        # Visual Detections
        # Run inference
        if visual:
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            t0 = time.time()
            # Padded resize
            img = letterbox(visual, img_size, stride=stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            t2 = time_synchronized()

            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], visual.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, visual, label=label, color=colors[int(cls)], line_thickness=3)

        concat_img = None
        if visual and thermal:
            concat_img = np.concatenate((thermal, visual), axis=1)
            concat_img = cv2.resize(concat_img, (1920, 720))

        elif visual:
            concat_img = visual

        elif thermal:
            concat_img = thermal

        cv2.imshow('Inferred Frame', concat_img)

        if cv2.waitKey(1) == 27:
            exit(0)


# parser = argparse.ArgumentParser()
# parser.add_argument('--weights', nargs='+', type=str, default='yolov5m_best_incense.pt', help='model.pt path(s)')
# parser.add_argument('--source', type=str, default='incense_yi.MOV', help='source')  # file/folder, 0 for webcam
# parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
# parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
# parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
# parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
# parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# parser.add_argument('--augment', action='store_true', help='augmented inference')
# opt = parser.parse_args()
#
# thermal_aspect = {'width': 4, 'height': 3}
# visual_aspect = {'width': 4, 'height': 3}
#
# device = select_device('0')  # 0 or 0,1,2,3 or cpu'
# half = device.type != 'cpu'  # half precision only supported on CUDA
# model = attempt_load('yolov5m_best_incense.pt', map_location=device)  # load FP32 model
# stride = int(model.stride.max())  # model stride
# imgsz = check_img_size(640, s=stride)  # check img_size
# if half:
#     model.half()  # to FP16
#
# names = model.module.names if hasattr(model, 'module') else model.names
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#
# webcam = cv2.VideoCapture('incense_yi.MOV')
#
# while True:
#     ret, frame = webcam.read()
#     # cv2.imshow('OG', frame)
#
#     thermal, visual = crop_image(frame, thermal_aspect, visual_aspect)
#     # Thermal Stream
#     thermal, tot_area, num_hotspots = thermal_detect(thermal)
#
#     # Visual Detections
#     visual = frame
#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     t0 = time.time()
#     # Padded resize
#     img = letterbox(visual, 640, stride=stride)[0]
#
#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)
#
#     img = torch.from_numpy(img).to(device)
#     img = img.half() if half else img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#
#     # Inference
#     t1 = time_synchronized()
#     pred = model(img, augment=opt.augment)[0]
#
#     # Apply NMS
#     pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#     t2 = time_synchronized()
#
#     for i, det in enumerate(pred):  # detections per image
#         gn = torch.tensor(visual.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#         if len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], visual.shape).round()
#
#             # Write results
#             for *xyxy, conf, cls in reversed(det):
#                 label = f'{names[int(cls)]} {conf:.2f}'
#                 plot_one_box(xyxy, visual, label=label, color=colors[int(cls)], line_thickness=3)
#
#     # concat_img = np.concatenate((thermal, visual), axis=1)
#     # concat_img = cv2.resize(concat_img, (1920, 720))
#     cv2.imshow('Inferred Frame', visual)
#
#     if cv2.waitKey(1) == 27:
#         exit(0)
