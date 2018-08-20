# RGB -> Gray

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
## collect cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

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

    if debug == "print":
        print_image(channels[channel], str(device) + "_rgb_" + names[channel] + ".png")
    elif debug == "plot":
        plot_image(channels[channel], cmap="gray")

    return device, channels[channel]


