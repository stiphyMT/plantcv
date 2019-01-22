# Scharr filtering

import cv2
import os
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import params
from plantcv.plantcv import PCVconstants as pcvc


def scharr_filter(img, dx, dy, scale):
    """This is a filtering method used to identify and highlight gradient edges/features using the 1st derivative.
       Typically used to identify gradients along the x-axis (dx = 1, dy = 0) and y-axis (dx = 0, dy = 1) independently.
       Performance is quite similar to Sobel filter. Used to detect edges / changes in pixel intensity. ddepth = -1
       specifies that the dimensions of output image will be the same as the input image.

    Inputs:
    gray_img = Grayscale image data
    dx       = derivative of x to analyze (1-3)
    dy       = derivative of x to analyze (1-3)
    scale    = scaling factor applied (multiplied) to computed Scharr values (scale = 1 is unscaled)

    Returns:
    sr_img   = Scharr filtered image

    :param img: numpy.ndarray
    :param dx: int
    :param dy: int
    :param scale: int
    :return sr_img: numpy.ndarray
    """

    sr_img = cv2.Scharr(src = img, ddepth = -1, dx = dx, dy = dy, scale = scale)
    params.device += 1
    if params.debug == pcvc.DEBUG_PRINT:
        name = os.path.join(params.debug_outdir, str(params.device))
        name = '{0}_sr_img_dx{1}_dy{2}_scale{3}.png'.format( name, dx, dy, scale)
        #name += '_sr_img_dx' + str(dx) + '_dy' + str(dy) + '_scale' + str(scale) + '.png'
        print_image(sr_img, name)
    elif params.debug == pcvc.DEBUG_PLOT:
        plot_image( sr_img, cmap='gray')
    return sr_img
