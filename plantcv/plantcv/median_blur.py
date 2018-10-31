# Median blur device

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import PCVconstants as pcvc


def median_blur(img, ksize, device, debug=None):
    """Applies a median blur filter (applies median value to central pixel within a kernel size ksize x ksize).

    Inputs:
    # img     = img object
    # ksize   = kernel size => ksize x ksize box
    # device  = device number. Used to count steps in the pipeline
    # debug   = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device    = device number
    img_mblur = blurred image

    :param img: numpy array
    :param ksize: int
    :param device: int
    :param debug: str
    :return device: int
    :return img_mblur: numpy array
    """
    if type(ksize) is int and ksize%2 == 1:
        fatal_error( "Invalid ksize, must be an odd number")
    elif type(ksize) is tuple and (ksize[0]%2 == 0 or ksize[1]%2 == 0):
        fatal_error( "Invalid ksize, both dimensions must be odd")
    elif type(ksize) is not int and type(ksize) is not tuple:
        fatal_error("Invalid ksize, must be integer or tuple")
    
    img_mblur = median_filter(gray_img, size=ksize)
    params.device += 1
    if debug == pcvc.DEBUG_PRINT:
        print_image(img_mblur, (str(device) + '_median_blur' + str(ksize) + '.png'))
    elif debug == pcvc.DEBUG_PLOT:
        plot_image(img_mblur, cmap='gray')
    return device, img_mblur
