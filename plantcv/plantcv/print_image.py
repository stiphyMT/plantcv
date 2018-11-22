# Print image to file
import sys
import cv2
from plantcv.plantcv import fatal_error
from plantcv.plantcv import PCVconstants as pcvc


def print_image(img, filename):
    """Save image to file.

    Inputs:
    img      = image object
    filename = name of file to save image to

    :param img: numpy.ndarray
    :param filename: string
    :return:
    """

    try:
        cv2.imwrite( filename, img)
    except:
        fatal_error( "Error writing file {0} : {1}".format( filename, sys.exc_info()[0]))
