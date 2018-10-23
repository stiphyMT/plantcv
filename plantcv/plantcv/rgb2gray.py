# RGB -> Gray

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import PCVconstants as pcvc

def rgb2gray(img, device, debug=None):
    """Convert image from RGB colorspace to Gray.

    Inputs:
    img    = image object, RGB colorspace
    device = device number. Used to count steps in the pipeline
    debug  = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device = device number
    gray   = grayscale image

    :param img: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return gray: numpy array
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    device += 1
    if debug == pcvc.DEBUG_PRINT:
        print_image(gray, (str(device) + '_gray.png'))
    elif debug == pcvc.DEBUG_PLOT:
        plot_image(gray, cmap='gray')
    return device, gray
