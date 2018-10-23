# Histogram equalization

import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error
from plantcv.plantcv import PCVconstants as pcvc

def hist_equalization(img, device, debug=None):
    """Histogram equalization is a method to normalize the distribution of intensity values. If the image has low
       contrast it will make it easier to threshold.

    Inputs:
    img    = input image
    device = device number. Used to count steps in the pipeline
    debug  = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device = device number
    img_eh = normalized image

    :param img: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return img_eh: numpy array
    """

    if len(np.shape(img)) == 3:
        fatal_error("Input image must be gray")

    img_eh = cv2.equalizeHist(img)
    device += 1
    if debug == pcvc.DEBUG_PRINT:
        print_image(img_eh, str(device) + '_hist_equal_img.png')
    elif debug == pcvc.DEBUG_PLOT:
        plot_image(img_eh, cmap = 'gray')

    return device, img_eh
