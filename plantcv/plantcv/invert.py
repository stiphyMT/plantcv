# Invert gray image

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image

## collect cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

def invert(img, device, debug=None):
    """Inverts grayscale images.

    Inputs:
    img     = image object, grayscale
    device  = device number. Used to count steps in the pipeline
    debug   = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device  = device number
    img_inv = inverted image

    :param img: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return img_inv: numpy array
    """

    device += 1
    img_inv = cv2.bitwise_not(img)
    if debug == 'print':
        print_image(img_inv, (str(device) + '_invert.png'))
    elif debug == 'plot':
        plot_image(img_inv, cmap='gray')
    return device, img_inv
