# Median blur device

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image

## collect cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

def median_blur(img, ksize, device, debug=None):
    """Applies a median blur filter (applies median value to central pixel within a kernel size ksize x ksize).

    Inputs:
    # img     = img object
    # ksize   = kernel size => ksize x ksize box
    # device  = device number. Used to count steps in the pipeline
    # debug   = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device    = device number
    img_mblur = blurred image

    :param img: numpy array
    :param ksize: int
    :param device: int
    :param debug: str
    :return device: int
    :return img_mblur: numpy array
    """

    img_mblur = cv2.medianBlur(img, ksize)
    device += 1
    if debug == 'print':
        print_image(img_mblur, (str(device) + '_median_blur' + str(ksize) + '.png'))
    elif debug == 'plot':
        plot_image(img_mblur, cmap='gray')
    return device, img_mblur
