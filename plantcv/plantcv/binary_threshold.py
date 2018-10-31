# Binary image threshold device

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error
from plantcv.plantcv import PCVconstants as pcvc

def binary_threshold(img, threshold, maxValue, object_type, device, debug=None):
    """Creates a binary image from a gray image based on the threshold value.

    Inputs:
    img         = img object, grayscale
    threshold   = threshold value (0-255)
    maxValue    = value to apply above threshold (usually 255 = white)
    object_type = light or dark
                  - If object is light then standard thresholding is done
                  - If object is dark then inverse thresholding is done
    device      = device number. Used to count steps in the pipeline
    debug       = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device      = device number
    t_img       = thresholded image

    :param img: numpy array
    :param threshold: int
    :param maxValue: int
    :param object_type: str
    :param device: int
    :param debug: str
    :return device: int
    :return t_img: numpy array
    """

    device += 1
    if object_type == pcvc.THRESHOLD_OBJ_LIGHT:
        ret, t_img = cv2.threshold(img, threshold, maxValue, cv2.THRESH_BINARY)
        if debug == pcvc.DEBUG_PRINT:
            print_image(t_img, (str(device) + '_binary_threshold' + str(threshold) + '.png'))
        elif debug == pcvc.DEBUG_PLOT:
            plot_image(t_img, cmap='gray')
        return device, t_img
    elif object_type == pcvc.THRESHOLD_OBJ_DARK:
        ret, t_img = cv2.threshold(img, threshold, maxValue, cv2.THRESH_BINARY_INV)
        if debug == pcvc.DEBUG_PRINT:
            print_image(t_img, (str(device) + '_binary_threshold' + str(threshold) + '_inv.png'))
        elif debug == pcvc.DEBUG_PLOT:
            plot_image(t_img, cmap='gray')
        return device, t_img
    else:
        fatal_error('Object type ' + str(object_type) + ' is not "light" or "dark"!')
