# RGB -> Gray

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error
from plantcv.plantcv import params
from plantcv.plantcv import PCVconstants as pcvc


def rgb2gray_rgb( rgb_img, channel):
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
    :return device: int
    :return channel: numpy array
    """
    # Split BGR channels
    params.device += 1
	
    # The allowable channel inputs are r, g or b
    names = { "r": "red", "g": "green", "b": "blue"}
	
    b, g, r = cv2.split( rgb_img)

    if channel not in names:
        fatal_error( "Channel " + str(channel) + " is not r, g or b!")

     # Create a channel dictionaries for lookups by a channel name index
    channels = { "r": r, "g": g, "b": b}

    if params.debug == pcvc.DEBUG_PRINT:
        print_image( channels[channel], '{0}_rgb_{1}.png'.format( params.device, names[channel]))
    elif params.debug == pcvc.DEBUG_PLOT:
        plot_image(channels[channel], cmap = pcvc.COLOUR_MAP_GREY)

    return channels[channel]


