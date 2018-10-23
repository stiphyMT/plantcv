# Binary image auto threshold

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import PCVconstants as pcvc


def otsu_auto_threshold(img, maxValue, object_type, device, debug=None):
    """Creates a binary image from a grayscaled image using Otsu's thresholding.

    Inputs:
    img         = img object, grayscale
    maxValue    = value to apply above threshold (usually 255 = white)
    object_type = light or dark
                  - If object is light then standard thresholding is done
                  - If object is dark then inverse thresholding is done
    device      = device number. Used to count steps in the pipeline
    debug       = True/False. If True, print image

    Returns:
    device      = device number
    t_img       = the thresholded image


    :param img: numpy array
    :param maxValue: int
    :param object_type: str
    :param device: int
    :param debug: bool
    :return device: int
    :return t_img: numpy array
    """
    device += 1

    # check whether to inverse the image or not and make an ending extension
    obj = 0
    extension = ''
    if object_type == pcvc.THRESHOLD_OBJ_LIGHT:
        extension = '.png'
        obj = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    elif object_type == pcvc.THRESHOLD_OBJ_DARK:
        extension = '_inv.png'
        obj = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    # threshold the image based on the object type using otsu's binarization
    t_val, t_img = cv2.threshold(img, 0, maxValue, obj)

    if debug == pcvc.DEBUG_PRINT:
        name = str(device) + '_otsu_auto_threshold_' + str(t_val) + str(extension)
        print_image(t_img, name)
    elif debug == pcvc.DEBUG_PLOT:
        plot_image(t_img, cmap="gray")

    return device, t_img
