# Watershed boundry detection function

import cv2
from . import print_image
from . import plot_image
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(major), int(minor)


def watershed(img, marker, device, debug=None):
    """Uses the watershed algorithm to detect boundry of objects. Needs a marker file which specifies area which is
       object (white), background (grey), unknown area (black).

    Inputs:
    img    = image to perform watershed on needs to be 3D (i.e. np.shape = x,y,z not np.shape = x,y)
    marker = a 32-bit image file that specifies what areas are what (2D, np.shape = x,y)
    device = device number. Used to count steps in the pipeline
    debug  = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device = device number
    marker = watershed filtered image

    :param img: numpy array
    :param marker: numpy array
    :param device: int
    :param debug: str
    :return device: int
    :return marker: numpy array
    """

    cv2.watershed(img, marker)
    device += 1
    if debug == 'print':
        print_image(marker, str(device) + 'watershed_img' + '.png')
    elif debug == 'plot':
        plot_image(marker, cmap='gray')
    return device, marker
