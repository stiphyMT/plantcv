# Histogram equalization

import cv2
from . import print_image
from . import plot_image
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(cv2major), int(cv2minor)


def HistEqualization(img, device, debug=None):
    """Histogram equalization is a method to normalize the distribution of intensity values. If the image has low
       contrast it will make it easier to threshold.

    Inputs:
    img    = input image
    device = device number. Used to count steps in the pipeline
    debug  = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device = device number
    img_eh = normalized image

    :param img: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return img_eh: numpy array
    """

    img_eh = cv2.equalizeHist(img)
    device += 1
    if debug == 'print':
        print_image(img_eh, str(device) + '_hist_equal_img.png')
    elif debug == 'plot':
        plot_image(img_eh)
    return device, img_eh
