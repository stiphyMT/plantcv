# Find NIR image

import os
import re
import numpy as np


def get_nir(path, filename, device, debug=None):
    """Find a corresponding NIR image from the same snapshot as the VIS image.

    Inputs:
    path     = path to vis image
    filename = vis image file name
    device   = pipeline step counter
    debug    = None, print, or plot. Print = save to file, Plot = print to screen.

    Returns:
    device   = device number
    nirpath  = NIR image filename and path

    :param path: str
    :param filename: str
    :param device: int
    :param debug: str
    :return device: int
    :return nirpath: str
    """

    device += 1
    visname = filename.split("_")
    allfiles = np.array(os.listdir(path))
    nirfiles = []

    targetimg = []
    cam = visname[1]

    if cam == "SV":
        angle = visname[2]

    for n in allfiles:
        if re.search("NIR", n) != None:
            nirfiles.append(n)

    if cam == "TV":
        for n in nirfiles:
            if re.search("TV", n) != None:
                nirpath = str(path) + "\\" + str(n)

    if cam == "SV":
        for n in nirfiles:
            if re.search("SV", n) != None:
                nsplit = n.split("_")
                exangle = '\\b' + str(angle) + '\\b'
                if re.search(exangle, nsplit[2]) != None:
                    nirpath = str(path) + "\\" + str(n)

    return device, nirpath
