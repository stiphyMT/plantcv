# Print image to file
import sys
import cv2
from . import fatal_error
## collet cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

def print_image(img, filename):
    """Save image to file.

    Inputs:
    img      = image object
    filename = name of file to save image to

    :param img: numpy array
    :param filename: string
    :return:
    """

    try:
        cv2.imwrite(filename, img)
    except:
        fatal_error("Unexpected error: " + str(sys.exc_info()[0]))
