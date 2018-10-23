# Gaussian blur device

import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import PCVconstants as pcvc

def gaussian_blur(device, img, ksize, sigmax=0, sigmay=None, debug=None):
    """Applies a Gaussian blur filter.

    Inputs:
    # device  = device number. Used to count steps in the pipeline
    # img     = img object
    # ksize   = kernel size => ksize x ksize box, e.g. (5,5)
    # sigmax = standard deviation in X direction; if 0, calculated from kernel size
    # sigmay = standard deviation in Y direction; if sigmaY is None, sigmaY is taken to equal sigmaX
    # debug   = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device    = device number
    img_gblur = blurred image

    :param img: numpy array
    :param ksize: tuple
    :param sigmax: int
    :param sigmay: str or int
    :param device: int
    :param debug: str
    :return device: int
    :return img_gblur: numpy array
    """

    img_gblur = cv2.GaussianBlur(img, ksize, sigmax, sigmay)

    device += 1
    if debug == pcvc.DEBUG_PRINT:
        print_image(img_gblur, (str(device) + '_gaussian_blur.png'))
    elif debug == pcvc.DEBUG_PLOT:
        if len(img_gblur) == 3:
            plot_image(img_gblur)
        else:
            plot_image(img_gblur, cmap='gray')

    return device, img_gblur
