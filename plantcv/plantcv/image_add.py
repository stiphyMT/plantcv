# Image addition

import os
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import params
from plantcv.plantcv import PCVconstants as pcvc

def image_add(gray_img1, gray_img2):

    """This is a function used to add images. The numpy addition function '+' is used. This is a modulo operation
       rather than the cv2.add fxn which is a saturation operation. ddepth = -1 specifies that the dimensions of output
       image will be the same as the input image.

    Inputs:
    gray_img1      = Grayscale image data to be added to image 2
    gray_img2      = Grayscale image data to be added to image 1

    Returns:
    added_img      = summed images

    :param gray_img1: numpy.ndarray
    :param gray_img2: numpy.ndarray
    :return added_img: numpy.ndarray
    """

    added_img = gray_img1 + gray_img2
    params.device += 1
    if params.debug == pcvc.DEBUG_PRINT:
        print_image( added_img, str( params.device) + '_added' + '.png')
    elif params.debug == pcvc.DEBUG_PLOT:
        plot_image( added_img, cmap = pcvc.COLOUR_MAP_GREY)
    return added_img
