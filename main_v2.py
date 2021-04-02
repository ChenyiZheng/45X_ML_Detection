import cv2
import numpy as np
import torch
import random
import time
from datetime import datetime
from working_scripts.utils import thermal_detect, plot_one_box, write_logs, time_synchronized, crop_image, save_frames
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov5m_best_incense.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='incense_yi.MOV', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args()


save_txt = 1
save_img = 1
filename = datetime.today().strftime('%Y-%m-%dT%H%M%S%z')

thermal_aspect = {'width': 4, 'height': 3}
visual_aspect = {'width': 4, 'height': 3}

# model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='yolov5m_best_incense.pt')  # custom model
device = '0'
device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA
model = attempt_load('yolov5m_best_incense.pt', map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
if half:
    model.half()  # to FP16

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# webcam = cv2.VideoCapture(0)

webcam = cv2.VideoCapture('incense_yi.MOV')
# webcam = cv2.VideoCapture(1)

while True:
    ret, frame = webcam.read()
    # cv2.imshow('OG', frame)

    thermal, visual = crop_image(frame, thermal_aspect, visual_aspect)
    # Thermal Stream
    thermal, tot_area, num_hotspots = thermal_detect(thermal)

    # Visual Detections

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    img = torch.from_numpy(visual).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    detection_results = np.array(pred.xyxy[0])

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
        else:
            visual_logs = 'None '

    concat_img = np.concatenate((thermal, visual), axis=1)
    concat_img = cv2.resize(concat_img, (1920, 720))
    cv2.imshow('Inferred Frame', concat_img)

    if cv2.waitKey(1) == 27:
        exit(0)