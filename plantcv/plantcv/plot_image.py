# Plot image to screen
import cv2
import numpy as np
from plantcv.plantcv import PCVconstants as pcvc


def plot_image( img, cmap = None):
    """Plot an image to the screen.

    :param img: numpy.ndarray
    :param cmap: str
    :return:
    """
    from matplotlib import pyplot as plt

    dimensions = np.shape( img)

    # If the image is color then OpenCV stores it as BGR, we plot it as RGB
    if len( dimensions) == 3:
        plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB))
        plt.show()
    elif cmap is None and len( dimensions) == 2:
        plt.imshow(img, cmap = pcvc.COLOUR_MAP_GREY)
        plt.show()
    elif cmap is not None and len( dimensions) == 2:
        plt.imshow( img, cmap = cmap)
        plt.show()
