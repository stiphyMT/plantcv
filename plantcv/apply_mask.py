# Apply White or Black Background Mask

import cv2
from . import print_image
from . import plot_image
from . import fatal_error
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(major), int(minor)


def apply_mask(img, mask, mask_color, device, debug=None):
    """Apply white image mask to image, with bitwise AND operator bitwise NOT operator and ADD operator.

    Inputs:
    img        = image object, color(RGB)
    mask       = image object, binary (black background with white object)
    mask_color = white or black
    device     = device number. Used to count steps in the pipeline
    debug      = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device     = device number
    masked_img = masked image

    :param img: numpy array
    :param mask: numpy array
    :param mask_color: str
    :param device: int
    :param debug: str
    :return device: int
    :return masked_img: numpy array
    """

    device += 1
    if mask_color == 'white':
        # Mask image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        # Create inverted mask for background
        mask_inv = cv2.bitwise_not(mask)
        # Invert the background so that it is white, but apply mask_inv so you don't white out the plant
        white_mask = cv2.bitwise_not(masked_img, mask=mask_inv)
        # Add masked image to white background (can't just use mask_inv because that is a binary)
        white_masked = cv2.add(masked_img, white_mask)
        if debug == 'print':
            print_image(white_masked, (str(device) + '_wmasked.png'))
        elif debug == 'plot':
            plot_image(white_masked)
        return device, white_masked
    elif mask_color == 'black':
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        if debug == 'print':
            print_image(masked_img, (str(device) + '_bmasked.png'))
        elif debug == 'plot':
            plot_image(masked_img)
        return device, masked_img
    else:
        fatal_error('Mask Color' + str(mask_color) + ' is not "white" or "black"!')
