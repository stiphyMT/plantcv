# Make masking rectangle

import cv2
import numpy as np
from . import print_image
from . import plot_image
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(major), int(minor)


def rectangle_mask(img, p1, p2, device, debug=None, color="black"):
    """Takes an input image and returns a binary image masked by a rectangular area denoted by p1 and p2. Note that
       p1 = (0,0) is the top left hand corner bottom right hand corner is p2 = (max-value(x), max-value(y)).

    Inputs:
    img       = image object
    p1        = point 1
    p2        = point 2
    device    = device number. Used to count steps in the pipeline
    debug     = None, print, or plot. Print = save to file, Plot = print to screen.
    color     = black or gray

    Returns:
    device    = device number
    bnk       = masked image
    contour   = object contour vertices
    hierarchy = contour hierarchy list

    :param img: numpy array
    :param p1: tuple
    :param p2: tuple
    :param device: int
    :param debug: str
    :param color: str
    :return device: int
    :return bnk: numpy array
    :return contour: list
    :return hierarchy: list
    """

    # get the dimensions of the input image
    ix, iy = np.shape(img)
    size = ix, iy
    # create a blank image of same size
    bnk = np.zeros(size, dtype=np.uint8)
    # draw a rectangle denoted by pt1 and pt2 on the blank image

    if color == "black":
        cv2.rectangle(img=bnk, pt1=p1, pt2=p2, color=(255, 255, 255))
        ret, bnk = cv2.threshold(bnk, 127, 255, 0)
        if cv2major > 2 and cv2minor > 0:
            _, contour, hierarchy = cv2.findContours(bnk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            contour, hierarchy = cv2.findContours(bnk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # make sure entire rectangle is within (visable within) plotting region or else it will not fill with
        # thickness = -1. Note that you should only print the first contour (contour[0]) if you want to fill with
        # thickness = -1. otherwise two rectangles will be drawn and the space between them will get filled
        cv2.drawContours(bnk, contour, 0, (255, 255, 255), -1)
        device += 1
    if color == "gray":
        cv2.rectangle(img=bnk, pt1=p1, pt2=p2, color=(192, 192, 192))
        ret, bnk = cv2.threshold(bnk, 127, 255, 0)
        if cv2major > 2 and cv2minor > 0:
            _, contour, hierarchy = cv2.findContours(bnk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            contour, hierarchy = cv2.findContours(bnk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # make sure entire rectangle is within (visable within) plotting region or else it will not fill with
        # thickness = -1. Note that you should only print the first contour (contour[0]) if you want to fill with
        # thickness = -1. otherwise two rectangles will be drawn and the space between them will get filled
        cv2.drawContours(bnk, contour, 0, (192, 192, 192), -1)
    if debug == 'print':
        print_image(bnk, (str(device) + '_roi.png'))
    elif debug == 'plot':
        plot_image(bnk)
    return device, bnk, contour, hierarchy
