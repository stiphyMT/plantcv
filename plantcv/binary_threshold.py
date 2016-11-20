# Binary image threshold device

import cv2
from . import print_image
from . import plot_image
from . import fatal_error
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(major), int(minor)


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
    if object_type == 'light':
        ret, t_img = cv2.threshold(img, threshold, maxValue, cv2.THRESH_BINARY)
        if debug == 'print':
            print_image(t_img, (str(device) + '_binary_threshold' + str(threshold) + '.png'))
        elif debug == 'plot':
            plot_image(t_img, cmap='gray')
        return device, t_img
    elif object_type == 'dark':
        ret, t_img = cv2.threshold(img, threshold, maxValue, cv2.THRESH_BINARY_INV)
        if debug == 'print':
            print_image(t_img, (str(device) + '_binary_threshold' + str(threshold) + '_inv.png'))
        elif debug == 'plot':
            plot_image(t_img, cmap='gray')
        return device, t_img
    else:
        fatal_error('Object type ' + str(object_type) + ' is not "light" or "dark"!')
