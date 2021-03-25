import cv2
import numpy as np


def thermal_detect(image, lower_bound=(0, 0, 127), upper_bound=(0, 0, 255)):
    """
    Draws a contour around the object of interest that is within the range passed.

    :param image: The original thermal image
    :param lower_bound: The lower bound of greyscale in HSV
    :param upper_bound: The upper bound of greyscale in HSV
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

    # print(area)
    # cv2.imshow('mask', mask)
    # cv2.imshow('original', original)
    # cv2.imshow('opening', opening)
    # cv2.waitKey()
    return original, area, len(cnts)


frame = cv2.imread('ThermVisArmImage.jpg')
original, area, length = thermal_detect(frame, lower_bound=[0, 0, 127])
cv2.imshow('OG', original)
cv2.waitKey()

# webcam = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = webcam.read()
#     thermal_detect(frame, lower_bound=[0, 0, 160])
