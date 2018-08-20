# Resize image

import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image

## collect cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

def auto_crop(device, img, objects, padding_x=0, padding_y=0, color='black', debug=None):
    """Resize image.

    Inputs:
    device    = device counter
    img       = image
    objects   = contours
    padding_x = padding in the x direction
    padding_y = padding in the y direction
    color     = either 'black' or 'white'
    debug     = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device    = device number
    cropped   = cropped image

    :param device: int
    :param img: numpy array
    :param objects: list
    :param padding_x: int
    :param padding_y: int
    :param color: str
    :param debug: str
    :return device: str
    :return cropped: numpy array
    """

    device += 1
    img_copy = np.copy(img)

    x, y, w, h = cv2.boundingRect(objects)
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 5)

    crop_img = img[y:y + h, x:x + w]

    offsetx = int(np.rint((padding_x)))
    offsety = int(np.rint((padding_y)))

    if color == 'black':
        colorval = (0, 0, 0)
    elif color == 'white':
        colorval = (255, 255, 255)

    cropped = cv2.copyMakeBorder(crop_img, offsety, offsety, offsetx, offsetx, cv2.BORDER_CONSTANT, value=colorval)

    if debug == 'print':
        print_image(img_copy, (str(device) + "_crop_area.png"))
        print_image(cropped, (str(device) + "_auto_cropped.png"))
    elif debug == 'plot':
        if len(np.shape(img_copy)) == 3:
            plot_image(img_copy)
            plot_image(cropped)
        else:
            plot_image(img_copy, cmap='gray')
            plot_image(cropped, cmap='gray')

    return device, cropped
