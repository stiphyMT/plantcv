# Join images (OR)

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import PCVconstants as pcvc


def logical_or(img1, img2, device, debug=None):
    """Join two images using the bitwise OR operator.

    Inputs:
    img1   = image object1, grayscale
    img2   = image object2, grayscale
    device = device number. Used to count steps in the pipeline
    debug  = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device = device number
    merged = joined image

    :param img1: numpy array
    :param img2: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return merged: numpy array
    """

    device += 1
    merged = cv2.bitwise_or(img1, img2)
    if debug == pcvc.DEBUG_PRINT:
        print_image(merged, (str(device) + '_or_joined.png'))
    elif debug == pcvc.DEBUG_PLOT:
        plot_image(merged, cmap='gray')
    return device, merged
