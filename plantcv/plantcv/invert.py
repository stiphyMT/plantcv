# Invert gray image

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import PCVconstants as pcvc


def invert(img, device, debug=None):
    """Inverts grayscale images.

    Inputs:
    img     = image object, grayscale
    device  = device number. Used to count steps in the pipeline
    debug   = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device  = device number
    img_inv = inverted image

    :param img: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return img_inv: numpy array
    """

    device += 1
    img_inv = cv2.bitwise_not(img)
    if debug == pcvc.DEBUG_PRINT:
        print_image(img_inv, (str(device) + '_invert.png'))
    elif debug == pcvc.DEBUG_PLOT:
        plot_image(img_inv, cmap='gray')
    return device, img_inv
