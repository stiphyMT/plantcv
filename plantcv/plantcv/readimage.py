# Read image

import os
import cv2
import numpy as np
from plantcv.plantcv import fatal_error
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import params
from plantcv.plantcv import PCVconstants as pcvc


def readimage(filename, mode = "native"):
    """Read image from file.

    Inputs:
    filename = name of image file
    mode     = mode of imread ("native", "rgb", "rgba", "gray")

    Returns:
    img      = image object as numpy array
    path     = path to image file
    img_name = name of image file

    :param filename: str
    :param mode: str
    :return img: numpy.ndarray
    :return path: str
    :return img_name: str
    """
    if mode.upper() == "GRAY" or mode.upper() == pcvc.READ_IMAGE_GRAY:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    elif mode.upper() == pcvc.READ_IMAGE_RGB:
        img = cv2.imread(filename)
    elif mode.upper() == pcvc.READ_IMAGE_RGBA:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread( filename, cv2.IMREAD_UNCHANGED)

    # Default to drop alpha channel if user doesn't specify 'rgba'
    if len(np.shape(img))==3 and np.shape(img)[2] == 4 and mode.upper() == "NATIVE":
        img = cv2.imread(filename)

    if img is None:
        fatal_error( "Failed to open " + filename)

    # Split path from filename
    path, img_name = os.path.split( filename)

    if params.debug == pcvc.DEBUG_PRINT:
        print_image( img, "input_image.png")
    elif params.debug == pcvc.DEBUG_PLOT:
        plot_image( img)

    return img, path, img_name
