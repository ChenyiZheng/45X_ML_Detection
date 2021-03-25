import cv2


def crop_image(image, thermal_aspect=None, visual_aspect=None):
    """
    A function to crop the image to two separate images. Assumes that half the width is dedicate to each image.
    :param image: The image
    :param thermal_aspect: The aspect ratio of the left most image
    :param visual_aspect: The aspect ratio of the right most image
    :return:
    """
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


# image = cv2.imread(r'C:\Users\imhen\PycharmProjects\45X_ML_Detection\thermalvisualvideo.mp4')
# thermal, visual = crop_image(image, thermal_aspect={'width': 4, 'height': 3}, visual_aspect={'width': 4, 'height': 3})
# cv2.imshow('Thermal', thermal)
# cv2.imshow('Visual', visual)
# cv2.waitKey()

webcam = cv2.VideoCapture(r'C:\Users\imhen\PycharmProjects\45X_ML_Detection\thermalvisualvideo.mp4')

while True:
    try:
        ret, frame = webcam.read()
        thermal, visual = crop_image(frame, thermal_aspect={'width': 4, 'height': 3}, visual_aspect={'width': 4, 'height': 3})
        cv2.imshow('Thermal', thermal)
        cv2.imshow('Visual', visual)
        cv2.waitKey(1)
    except:
        cv2.destroyAllWindows()
        break
