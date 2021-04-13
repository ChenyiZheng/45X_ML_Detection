import cv2
import numpy as np
import random
import os
from working_scripts.utils import crop_image


def save_frames(filename, original_frame, num):
    root_dir = os.path.abspath(os.curdir)
    img_dir = f'{root_dir}/logs/images/{filename}'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cv2.imwrite(os.path.join(img_dir, f'fireplace_0{num}.jpg'), original_frame)


directory = r'C:\Users\imhen\Pictures\Camera Roll\fireplace_pictures'
video = cv2.VideoCapture(r'C:\Users\imhen\Pictures\Camera Roll\Smoke_2.mkv')

num = 19
thermal_aspect = {'width': 4, 'height': 3}
visual_aspect = {'width': 4, 'height': 3}

exclude_p = 0.01

while True:
    ret, frame = video.read()
    thermal, visual = crop_image(frame, thermal_aspect, visual_aspect)
    rand = random.random()
    if exclude_p >= rand:
        num += 1
        cv2.imwrite(os.path.join(directory, f'smoke_0{num}.jpg'), visual)
