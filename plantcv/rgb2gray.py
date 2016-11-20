# RGB -> Gray

import cv2
from . import print_image
from . import plot_image
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(cv2major), int(cv2minor)


def rgb2gray(img, device, debug=None):
    """Convert image from RGB colorspace to Gray.

    Inputs:
    img    = image object, RGB colorspace
    device = device number. Used to count steps in the pipeline
    debug  = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device = device number
    gray   = grayscale image

    :param img: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return gray: numpy array
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    device += 1
    if debug == 'print':
        print_image(gray, (str(device) + '_gray.png'))
    elif debug == 'plot':
        plot_image(gray, cmap='gray')
    return device, gray
