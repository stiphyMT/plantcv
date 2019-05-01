# Watershed Se detection function
# This function is based on code contributed by Suxing Liu, Arkansas State University.
# For more information see https://github.com/lsx1980/Leaf_count

import cv2
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import apply_mask
from plantcv.plantcv import color_palette
from plantcv.plantcv import params
from plantcv.plantcv import outputs
from plantcv.plantcv import PCVconstants as pcvc


def watershed_segmentation(rgb_img, mask, distance = 10):
    """Uses the watershed algorithm to detect boundary of objects. Needs a marker file which specifies area which is
       object (white), background (grey), unknown area (black).

    Inputs:
    rgb_img             = image to perform watershed on needs to be 3D (i.e. np.shape = x,y,z not np.shape = x,y)
    mask                = binary image, single channel, object in white and background black
    distance            = min_distance of local maximum

    Returns:
    watershed_header    = shape data table headers
    watershed_data      = shape data table values
    analysis_images     = list of output images

    :param rgb_img: numpy.ndarray
    :param mask: numpy.ndarray
    :param distance: int
    :return watershed_header: list
    :return watershed_data: list
    :return analysis_images: list
    """
    params.device += 1
    if pcvc.CV2MAJOR >= 3:
        dist_transform, _ = cv2.distanceTransformWithLabels( mask, cv2.DIST_L2, maskSize = 0)
    else:
        dist_transform = cv2.distanceTransform( mask, cv2.cv.CV_DIST_L2, maskSize = 0)
==== BASE ====

    localMax = peak_local_max( dist_transform, indices = False, min_distance = distance, labels = mask)

    markers = ndi.label( localMax, structure = np.ones(( 3, 3)))[0]
    dist_transform1 = -dist_transform
    labels = watershed( dist_transform1, markers, mask = mask)

    img1 = np.copy( rgb_img)

    for x in np.unique( labels):
	# should this be here or just before the loop?
        rand_color = color_palette( len( np.unique(labels)))
        img1[labels == x] = rand_color[x]

    img2 = apply_mask( img1, mask, 'black')

    joined = np.concatenate(( img2, rgb_img), axis = 1)

    estimated_object_count = len( np.unique( markers)) - 1

    analysis_image = []
    analysis_image.append(joined)

    watershed_header = (
        'HEADER_WATERSHED',
        'estimated_object_count'
    )

    watershed_data = (
        'WATERSHED_DATA',
        estimated_object_count
    )

    if params.debug == pcvc.DEBUG_PRINT:
        print_image( dist_transform, str( params.device) + '_watershed_dist_img.png')
        print_image( joined, str( params.device) + '_watershed_img.png')
    elif params.debug == pcvc.DEBUG_PLOT:
        plot_image( dist_transform, cmap = pcvc.COLOUR_MAP_GREY)
        plot_image( joined)

    # Store into global measurements
    if not 'watershed' in outputs.measurements:
        outputs.measurements['watershed'] = {}
    outputs.measurements['watershed']['estimated_object_count'] = estimated_object_count

    # Store images
    outputs.images.append(analysis_image)

    return watershed_header, watershed_data, analysis_image
