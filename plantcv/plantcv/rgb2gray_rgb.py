# RGB -> Gray

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error
from plantcv.plantcv import PCVconstants as pcvc


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
    device += 1
	
    # The allowable channel inputs are h, s or v
    names = {"r": "red", "g": "green", "b": "blue"}
	
    b, g, r = cv2.split( img)

    if channel not in names:
        fatal_error("Channel " + str(channel) + " is not r, g or b!")

     # Create a channel dictionaries for lookups by a channel name index
    channels = {"r": r, "g": g, "b": b}

    if debug == pcvc.DEBUG_PRINT:
        print_image(channels[channel], str(device) + "_rgb_" + names[channel] + ".png")
    elif debug == pcvc.DEBUG_PLOT:
        plot_image(channels[channel], cmap="gray")

    return device, channels[channel]


