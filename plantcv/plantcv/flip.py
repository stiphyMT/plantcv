# Flip image

import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error
from plantcv.plantcv import PCVconstants as pcvc

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

    if debug == pcvc.DEBUG_PRINT:
        print_image(vh_img, (str(device) + "_flipped.png"))
    elif debug == pcvc.DEBUG_PLOT:
        if len(np.shape(vh_img)) == 3:
            plot_image(vh_img)
        else:
            plot_image(vh_img, cmap='gray')

    return device, vh_img
