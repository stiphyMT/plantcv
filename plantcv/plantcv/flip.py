# Flip image

import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error

## collect cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

def flip(img, direction, device, debug=None):
    """Flip image.

    Inputs:
    img       = image to be flipped
    direction = "horizontal" or "vertical"
    device    = device counter
    debug     = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device    = device number
    vh_img    = flipped image

    :param img: numpy array
    :param direction: str
    :param device: int
    :param debug: str
    :return device: int
    :return vh_img: numpy array
    """
    device += 1
    
    if direction == "vertical":
        vh_img = cv2.flip(img, 1)
    elif direction == "horizontal":
        vh_img = cv2.flip(img, 0)
    else:
        fatal_error(str(direction) + " is not a valid direction, must be horizontal or vertical")

    if debug == 'print':
        print_image(vh_img, (str(device) + "_flipped.png"))
    elif debug == 'plot':
        if len(np.shape(vh_img)) == 3:
            plot_image(vh_img)
        else:
            plot_image(vh_img, cmap='gray')

    return device, vh_img
