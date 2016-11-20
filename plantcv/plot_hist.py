# Plot histogram

import cv2
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(cv2major), int(cv2minor)



def plot_hist(img, name):
    """Plot a histogram using the pyplot library.

    Inputs:
    img  = image to analyze
    name = name for plot output

    :param img: numpy array
    :param name: str
    :return:
    """

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    # get histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    # open pyplot plotting window using hist data
    plt.plot(hist)
    # set range of x-axis
    xaxis = plt.xlim([0, (255)])
    fig_name = name + '.png'
    # write the figure to current directory
    plt.savefig(fig_name)
    # close pyplot plotting window
    plt.clf()
