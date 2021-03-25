import cv2
import numpy as np
import random


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
