# Sobel filtering

import cv2
import os
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import params
from plantcv.plantcv import PCVconstants as pcvc


def sobel_filter( gray_img, dx, dy, ksize):
    """This is a filtering method used to identify and highlight gradient edges/features using the 1st derivative.
       Typically used to identify gradients along the x-axis (dx = 1, dy = 0) and y-axis (dx = 0, dy = 1) independently.
       Performance is quite similar to Scharr filter. Used to detect edges / changes in pixel intensity. ddepth = -1
       specifies that the dimensions of output image will be the same as the input image.

    Inputs:
    gray_img = Grayscale image data
    dx       = derivative of x to analyze
    dy       = derivative of x to analyze
    ksize        = specifies the size of the kernel (must be an odd integer: 1,3,5, ... , 31)

    Returns:
    sb_img   = Sobel filtered image

    :param gray_img: numpy.ndarray
    :param dx: int
    :param dy: int
    :param ksize: int
    :param scale: int
    :return sb_img: numpy.ndarray
    """
    params.device += 1
    sb_img = cv2.Sobel( src = gray_img, ddepth = -1, dx = dx, dy = dy, ksize = ksize)

    if params.debug == pcvc.DEBUG_PRINT:
        name = os.path.join(params.debug_outdir,
                            " {0}_sb_img_dx{1}_dy{2}_kernel{3}.png".format( params.device, dx, dy, ksize))
        print_image( sb_img, name)
    elif params.debug == pcvc.DEBUG_PLOT:
        plot_image( sb_img, cmap = pcvc.COLOR_MAP_GRAY)
    return sb_img
