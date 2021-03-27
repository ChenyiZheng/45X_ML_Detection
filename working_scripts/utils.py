import cv2
import numpy as np
import random
import torch
import time
import os


def thermal_detect(image, lower_bound: int = (0, 0, 127), upper_bound: int = (0, 0, 255)):
    """
    Draws a contour around the object of interest that is within the range passed.

    :param image: The original thermal image
    :param lower_bound: The lower bound in HSV
    :param upper_bound: The upper bound in HSV
    :return:
    """
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV format
    lower = np.array(lower_bound, dtype="uint8")    # Arrays for lower and upper bounds of detection
    upper = np.array(upper_bound, dtype="uint8")
    mask = cv2.inRange(image, lower, upper)         # Create a mask using the image and bounds

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))      # Remove noise with morphological transformations
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours based on the mask
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    area = 0
    for c in cnts:
        area += cv2.contourArea(c)
        cv2.drawContours(original, [c], 0, (0, 220, 255), 2)    # Draw the contours on top of the original image

    return original, area, len(cnts)


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Draws a bounding box

    :param x: The coordinates for the box
    :param img: The original image
    :param color: The desired colour in BGR
    :param label: The label to we written with the rectangle
    :param line_thickness: The desired line thickness of the box
    :return:
    """
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


def write_logs(filename, num_hotspots, tot_area, visual_logs, timestamp, visual_processed_time):
    root_dir = os.path.abspath(os.curdir)
    logs_dir = f'{root_dir}/logs/'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    path = os.path.join(logs_dir, filename)
    with open(filename + '.txt', 'a') as f:
        if os.stat(filename + '.txt').st_size == 0:
            f.write('detection starts at ' + filename + '\n' +
                    "{:<13}".format('Time') + "{:<40}".format(' Thermal Results') + '  Visual Results \n')
        else:
            thermal_logs = f'# of hotspots: {num_hotspots}, total area: {tot_area}'
            line = f"{timestamp:<13} {thermal_logs:<40} {visual_logs}({visual_processed_time} s) Done. "
            f.write(line + '\n')


def save_frames(filename, timestamp, original_frame, processed_frame):
    root_dir = os.path.abspath(os.curdir)
    img_dir = f'{root_dir}/logs/images/{filename}'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    timestamp = timestamp.replace(":", "_")
    cv2.imwrite(os.path.join(img_dir, f'{timestamp}_original.jpg'), original_frame)
    cv2.imwrite(os.path.join(img_dir, f'{timestamp}_processed.jpg'), processed_frame)


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def crop_image(image, thermal_aspect=None, visual_aspect=None):
    (height, width) = image.shape[:2]
    half_width = int(round(width/2))
    thermal_coords = {'x0': 0,
                      'y0': 0,
                      'x1': half_width,
                      'y1': int(round(half_width*thermal_aspect['height']/thermal_aspect['width']))
                      }

    visual_coords = {'x0': thermal_coords['x1'],
                     'y0': 0,
                     'x1': int(round(thermal_coords['x1'] + width/2)),
                     'y1': int(round(half_width*visual_aspect['height']/visual_aspect['width']))
                     }

    thermal = image[thermal_coords['y0']:thermal_coords['y1'], thermal_coords['x0']:thermal_coords['x1']]
    visual = image[visual_coords['y0']:visual_coords['y1'], visual_coords['x0']:visual_coords['x1']]

    return thermal, visual