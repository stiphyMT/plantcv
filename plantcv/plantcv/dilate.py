# Dilation filter

import cv2
import numpy as np
import os
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import params
from plantcv.plantcv import PCVconstants as pcvc


def dilate(gray_img, kernel, i):
    """Performs morphological 'dilation' filtering. Adds pixel to center of kernel if conditions set in kernel are true.

    Inputs:
    gray_img = Grayscale (usually binary) image data
    kernel   = Kernel size (int). A k x k kernel will be built. Must be greater than 1 to have an effect.
    i        = interations, i.e. number of consecutive filtering passes

    Returns:
    dil_img = dilated image

    :param gray_img: numpy.ndarray
    :param kernel: int
    :param i: int
    :return dil_img: numpy.ndarray
    """

    kernel1 = int(kernel)
    kernel2 = np.ones((kernel1, kernel1), np.uint8)
    dil_img = cv2.dilate(src = gray_img, kernel = kernel2, iterations = i)
    params.device += 1
    if params.debug == pcvc.DEBUG_PRINT:
        print_image( dil_img, "{0}_dil_image_itr{1}.png".format( params.device, i))
    elif params.debug == pcvc.DEBUG_PLOT:
        plot_image(dil_img, cmap='gray')
    return dil_img
