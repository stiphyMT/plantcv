# RGB -> Gray

import cv2
from . import print_image
from . import plot_image


def rgb2gray_rgb(img, channel, device, debug=None):
    """Convert image from RGB colorspace to Gray.

    Inputs:
    img    = image object, RGB colorspace
    channel = color subchannel (r = red, g = green, b = bue)
    device = device number. Used to count steps in the pipeline
    debug  = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device = device number
    r | g | b = image from single RGB channel

    :param img: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return channel: numpy array
    """
    # Split BGR channels
    b, g, r = cv2.split( img)
    device += 1
    if channel == 'b':
        if debug == 'print':
            print_image(b, (str(device) + '_rgb_blue.png'))
        elif debug == 'plot':
            plot_image(b, cmap='gray')
        return device, b
    elif channel == 'g':
        if debug == 'print':
            print_image(g, (str(device) + '_rgb_green.png'))
        elif debug == 'plot':
            plot_image(g, cmap='gray')
        return device, g
    elif channel == 'r':
        if debug == 'print':
            print_image(r, (str(device) + '_rbg_red.png'))
        elif debug == 'plot':
            plot_image(r, cmap='gray')
        return device, r
    else:
        fatal_error('Channel ' + (str(channel) + ' is not b, g or r!'))
