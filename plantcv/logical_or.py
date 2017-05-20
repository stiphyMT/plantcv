# Join images (OR)

import cv2
from . import print_image
from . import plot_image
## collet cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

def logical_or(img1, img2, device, debug=None):
    """Join two images using the bitwise OR operator.

    Inputs:
    img1   = image object1, grayscale
    img2   = image object2, grayscale
    device = device number. Used to count steps in the pipeline
    debug  = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device = device number
    merged = joined image

    :param img1: numpy array
    :param img2: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return merged: numpy array
    """

    device += 1
    merged = cv2.bitwise_or(img1, img2)
    if debug == 'print':
        print_image(merged, (str(device) + '_or_joined.png'))
    elif debug == 'plot':
        plot_image(merged, cmap='gray')
    return device, merged
