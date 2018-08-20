# RGB -> LAB -> Gray

import cv2
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error

## collect cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

def rgb2gray_lab(img, channel, device, debug=None):
    """Convert image from RGB colorspace to LAB colorspace. Returns the specified subchannel as a gray image.

    Inputs:
    img       = image object, RGB colorspace
    channel   = color subchannel (l = lightness, a = green-magenta, b = blue-yellow)
    device    = device number. Used to count steps in the pipeline
    debug     = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device    = device number
    l | a | b = grayscale image from one LAB color channel

    :param img: numpy array
    :param channel: str
    :param device: int
    :param debug: str
    :return device: int
    :return channel: numpy array
    """
    # Auto-increment the device counter
    device += 1
    # The allowable channel inputs are l, a or b
    names = {"l": "lightness", "a": "green-magenta", "b": "blue-yellow"}
    if channel not in names:
        fatal_error("Channel " + str(channel) + " is not l, a or b!")

    # Convert the input BGR image to LAB colorspace
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Split LAB channels
    l, a, b = cv2.split(lab)
    # Create a channel dictionaries for lookups by a channel name index
    channels = {"l": l, "a": a, "b": b}

    if debug == "print":
        print_image(channels[channel], str(device) + "_lab_" + names[channel] + ".png")
    elif debug == "plot":
        plot_image(channels[channel], cmap="gray")

    return device, channels[channel]
