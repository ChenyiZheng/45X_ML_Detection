import cv2
import numpy as np
import torch
import random
import time, os
from datetime import datetime
from working_scripts.utils import thermal_detect, plot_one_box, write_logs, time_synchronized, crop_image, save_frames, \
    save_videos

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized


def detect(weights: str,
           source: str,
           device: str,
           img_size: int,
           conf_thres: float,
           iou_thres: float,
           classes: [int],
           agnostic_nms: str,
           augment: str,
           thermal_aspect=None,
           visual_aspect=None,
           save_video=False):
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
    fps = int(webcam.get(cv2.CAP_PROP_FPS))
    filename = datetime.today().strftime('%Y-%m-%dT%H%M%S%z')

    while True:
        ret, frame = webcam.read()

        thermal = None
        visual = None

        if thermal_aspect and visual_aspect:
            thermal, visual = crop_image(frame, thermal_aspect, visual_aspect)

        elif thermal_aspect:
            thermal = frame

        elif visual_aspect:
            visual = frame

        # Thermal Stream
        if thermal_aspect:
            thermal, tot_area, num_hotspots = thermal_detect(thermal)

        # Visual Detections
        # Run inference
        if visual_aspect:
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
        if thermal_aspect and visual_aspect:
            concat_img = np.concatenate((thermal, visual), axis=1)
            concat_img = cv2.resize(concat_img, (1920, 720))

        elif visual_aspect:
            concat_img = visual

        elif thermal_aspect:
            concat_img = thermal

        cv2.imshow('Inferred Frame', concat_img)

        if save_video:
            # save_videos(filename, concat_img, fps)
            (height, width) = frame.shape[:2]
            size = (width, height)
            root_dir = 'runs/'
            video_dir = f'{root_dir}\{filename}.avi'
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            if not os.path.exists(video_dir):
                os.makedirs('runs/')
                out_vid = cv2.VideoWriter(video_dir,
                                          cv2.VideoWriter_fourcc(*'MJPG'),
                                          fps, size)
            else:
                out_vid.write(frame)

        if cv2.waitKey(1) == 27:
            exit(0)


if __name__ == '__main__':
    thermal_aspect = {'width': 4, 'height': 3}
    visual_aspect = {'width': 4, 'height': 3}
    detect(weights='yolov5m_best_incense.pt',
           source='incense_yi.MOV',
           device='0',
           img_size=640,
           conf_thres=0.1,
           iou_thres=0.25,
           classes=[0, 1],
           agnostic_nms='store_true',
           augment='store_true',
           thermal_aspect=None,
           visual_aspect=visual_aspect,
           save_video=True)
