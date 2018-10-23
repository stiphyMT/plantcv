# line blur device

import cv2
import numpy as np
from skimage.filters import median as skmedian
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import PCVconstants as pcvc


def line_blur(img, kernel, device, direction = 0, debug=None):
    """Applies a median blur filter from skimage with rectangular kernel 
    (applies median value to central pixel within a kernel size ksize x 1, or 1 x ksize).
    This is used to remove thin horizontal and vertical lines

    Inputs:
    # img       = img object
    # ksize     = kernel size long dimension
    # direction = 0 for horizontal, 1 for vertical
    # device    = device number. Used to count steps in the pipeline
    # debug     = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device    = device number
    img_mblur = blurred image

    :param img: numpy array
    :param ksize: int
    :param direction: int (0 or 1)
    :param device: int
    :param debug: str
    :return device: int
    :return img_mblur: numpy array
    """
    
    if direction == 0:
        kern = np.ones( (kernel, 1), np.uint8)
    elif direction == 1:
        kern = np.ones( (1, kernel), np.uint8)
    else:
        print( "Direction selected was not 0, or 1") 
        return device, img
        
    img_mblur = skmedian(img, selem = kern)
    
    device += 1
    if debug == pcvc.DEBUG_PRINT:
#        print_image(img_mblur, (str(device) + '_line_blur' + str(kernel) + '.png'))
        print_image(img_mblur, ("{0}_line_blur_{1}.png".format( device, ((1, kernel), (kernel,1))[direction])))
    elif debug == pcvc.DEBUG_PLOT:
        plot_image(img_mblur, cmap = 'gray')
    return device, img_mblur
