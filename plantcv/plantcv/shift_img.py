# Crop position mask

import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error
from plantcv.plantcv import PCVconstants as pcvc


def shift_img(img, device, number, side="right", debug=None):
    """this function allows you to shift an image over without changing dimensions

    Inputs:
    img     = image to mask
    number  = number of rows or columns to add
    side   = "top", "bottom", "right", "left" where to add the rows or columns to
    device  = device counter
    debug   = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device  = device number
    newmask = image mask

    :param img: numpy array
    :param device: int
    :param number: int
    :param side: str
    :param debug: str
    :return newmask: numpy array
    """
    device += 1

    number = number - 1

    if number < 0:
        fatal_error("x and y cannot be negative numbers or non-integers")

    # get the sizes of the images
    if len(np.shape(img)) == 3:
        ix, iy, iz = np.shape(img)
        ori_img = np.copy(img)
    else:
        ix, iy = np.shape(img)
        ori_img = np.dstack((img, img, img))

    if side == pcvc.SHIFT_IMG_TOP:
        top = np.zeros((number, iy, 3), dtype=np.uint8)
        adjust = ix - number
        adjusted_img = np.vstack((top, ori_img[0:adjust, 0:]))

    if side == pcvc.SHIFT_IMG_BOTTOM:
        bottom = np.zeros((number, iy, 3), dtype=np.uint8)
        adjusted_img = np.vstack((ori_img[number:, 0:], bottom))

    if side == pcvc.SHIFT_IMG_RIGHT:
        right = np.zeros((ix, number, 3), dtype=np.uint8)
        adjusted_img = np.hstack((ori_img[0:, number:], right))
    if side == pcvc.SHIFT_IMG_LEFT:
        left = np.zeros((ix, number, 3), dtype=np.uint8)
        adjust = iy - number
        adjusted_img = np.hstack((left, ori_img[0:, 0:adjust]))

    if len(np.shape(img)) == 2:
        adjusted_img = adjusted_img[:,:,0]
    if debug == pcvc.DEBUG_PRINT:
        print_image(adjusted_img, (str(device) + "_shifted_img.png"))
    elif debug == pcvc.DEBUG_PLOT:
        if len(np.shape(adjusted_img)) == 3:
            plot_image(adjusted_img)
        else:
            plot_image(adjusted_img, cmap='gray')

    return device, adjusted_img
