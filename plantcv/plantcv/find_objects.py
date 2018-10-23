# Find Objects

import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import PCVconstants as pcvc

def find_objects(img, mask, device, debug=None):
    """Find all objects and color them blue.

    Inputs:
    img       = image that the objects will be overlayed
    mask      = what is used for object detection
    device    = device number.  Used to count steps in the pipeline
    debug     = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device    = device number
    objects   = list of contours
    hierarchy = contour hierarchy list

    :param img: numpy array
    :param mask: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return objects: list
    :return hierarchy: list
    """

    device += 1
    mask1 = np.copy(mask)
    ori_img = np.copy(img)
    if pcvc.CV2MAJOR >= 3 and pcvc.CV2MINOR >= 1:
        _, objects, hierarchy = cv2.findContours( mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        objects, hierarchy = cv2.findContours( mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate( objects):
        cv2.drawContours( ori_img, objects, i, ( 255, 102, 255), -1, lineType = 8, hierarchy = hierarchy)
    if debug == pcvc.DEBUG_PRINT:
        print_image( ori_img, ( str( device) + '_id_objects.png'))
    elif debug == pcvc.DEBUG_PLOT:
        plot_image( ori_img)

    return device, objects, hierarchy
