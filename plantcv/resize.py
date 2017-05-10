# Resize image

import cv2
from . import print_image
from . import plot_image
from . import fatal_error
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(cv2major), int(cv2minor)


def resize(img, resize_x, resize_y, device, debug=None):
    """Resize image.

    Inputs:
    img      = image to resize
    resize_x = scaling factor
    resize_y = scaling factor
    device   = device counter
    debug    = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device   = device number
    reimg    = resized image

    :param img: numpy array
    :param resize_x: int
    :param resize_y: int
    :param device: int
    :param debug: str
    :return device: int
    :return reimg: numpy array
    """

    device += 1

    reimg = cv2.resize(img, (0, 0), fx=resize_x, fy=resize_y)

    if resize_x <= 0 and resize_y <= 0:
        fatal_error("Resize values both cannot be 0 or negative values!")

    if debug == 'print':
        print_image(reimg, (str(device) + "_resize1.png"))
    elif debug == 'plot':
        plot_image(reimg, cmap='gray')

    return device, reimg
