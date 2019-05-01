# Analyzes an object and outputs numeric properties

import cv2
import numpy as np
import os
from plantcv.plantcv import fatal_error
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import rgb2gray_hsv
from plantcv.plantcv import find_objects
from plantcv.plantcv.threshold import binary as binary_threshold
from plantcv.plantcv import roi_objects
from plantcv.plantcv import object_composition
from plantcv.plantcv import apply_mask
from plantcv.plantcv import params
from plantcv.plantcv import outputs
from plantcv.plantcv import PCVconstants as pcvc

def report_size_marker_area( img, roi_contour, roi_hierarchy, marker = 'define', objcolor = pcvc.THRESHOLD_OBJ_DARK, thresh_channel = None,
                            thresh = None):
    """Detects a size marker in a specified region and reports its size and eccentricity

    Inputs:
    img             = An RGB or grayscale image to plot the marker object on
    roi_contour     = A region of interest contour (e.g. output from pcv.roi.rectangle or other methods)
    roi_hierarchy   = A region of interest contour hierarchy (e.g. output from pcv.roi.rectangle or other methods)
    marker          = 'define' or 'detect'. If define it means you set an area, if detect it means you want to
                      detect within an area
    objcolor        = Object color is 'dark' or 'light' (is the marker darker or lighter than the background)
    thresh_channel  = 'h', 's', or 'v' for hue, saturation or value
    thresh          = Binary threshold value (integer)

    Returns:
    marker_header   = Marker data table headers
    marker_data     = Marker data table values
    analysis_images = List of output images

    :param img: numpy.ndarray
    :param roi_contour: list
    :param roi_hierarchy: numpy.ndarray
    :param marker: str
    :param objcolor: str
    :param thresh_channel: str
    :param thresh: int
    :return: marker_header: list
    :return: marker_data: list
    :return: analysis_images: list
    """

    params.device += 1
    # Make a copy of the reference image
    ref_img = np.copy(img)
    # If the reference image is grayscale convert it to color
    if len( np.shape( ref_img)) == 2:
        ref_img = cv2.cvtColor( ref_img, cv2.COLOR_GRAY2BGR)

    # Marker components
    # If the marker type is "defined" then the marker_mask and marker_contours are equal to the input ROI
    # Initialize a binary image
    roi_mask = np.zeros( np.shape( img)[:2], dtype = np.uint8)
    # Draw the filled ROI on the mask
    cv2.drawContours( roi_mask, roi_contour, -1, (255), -1)
    marker_mask = []
    marker_contour = []

    # If the marker type is "detect" then we will use the ROI to isolate marker contours from the input image
    if marker.upper() == pcvc.REPORT_SIZE_MARKER_DETECT:
        # We need to convert the input image into an one of the HSV channels and then threshold it
        if thresh_channel is not None and thresh is not None:
            # Mask the input image
            masked = apply_mask( rgb_img = ref_img, mask = roi_mask, mask_color = pcvc.APPLY_MASK_BLACK)
            # Convert the masked image to hue, saturation, or value
            marker_hsv = rgb2gray_hsv( rgb_img = masked, channel = thresh_channel)
            # Threshold the HSV image
            marker_bin = binary_threshold( gray_img = marker_hsv, threshold = thresh, max_value = 255, object_type = objcolor)
            # Identify contours in the masked image
            contours, hierarchy = find_objects( img = ref_img, mask = marker_bin)
            # Filter marker contours using the input ROI
            kept_contours, kept_hierarchy, kept_mask, obj_area = roi_objects( img = ref_img, object_contour = contours,
                                                                             obj_hierarchy = hierarchy,
                                                                             roi_contour = roi_contour,
                                                                             roi_hierarchy = roi_hierarchy,
                                                                             roi_type = "partial")
            # If there are more than one contour detected, combine them into one
            # These become the marker contour and mask
            marker_contour, marker_mask = object_composition(img = ref_img, contours = kept_contours,
                                                             hierarchy = kept_hierarchy)
        else:
            fatal_error('thresh_channel and thresh must be defined in detect mode')
    elif marker.upper() == pcvc.REPORT_SIZE_MARKER_DEFINE:
        # Identify contours in the masked image
        contours, hierarchy = find_objects( img = ref_img, mask = roi_mask)
        # If there are more than one contour detected, combine them into one
        # These become the marker contour and mask
        marker_contour, marker_mask = object_composition( img = ref_img, contours = contours, hierarchy = hierarchy)
    else:
        fatal_error("marker must be either 'define' or 'detect' but {0} was provided.".format( marker))

    # Calculate the moments of the defined marker region
    m = cv2.moments( marker_mask, binaryImage = True)
    # Calculate the marker area
    marker_area = m['m00']

    # Fit a bounding ellipse to the marker
    center, axes, angle = cv2.fitEllipse( marker_contour)
    major_axis = np.argmax( axes)
    minor_axis = 1 - major_axis
    major_axis_length = axes[major_axis]
    minor_axis_length = axes[minor_axis]
    # Calculate the bounding ellipse eccentricity
    eccentricity = np.sqrt( 1 - (axes[minor_axis] / axes[major_axis]) ** 2)

    # Make a list to store output images
    analysis_image = []
    cv2.drawContours(ref_img, marker_contour, -1, (255, 0, 0), 5)
    # out_file = os.path.splitext(filename)[0] + '_sizemarker.jpg'
    # print_image(ref_img, out_file)
    analysis_image.append(ref_img)
    if params.debug is pcvc.DEBUG_PRINT:
        print_image(ref_img, os.path.join(params.debug_outdir, str(params.device) + '_marker_shape.png'))
    elif params.debug is pcvc.DEBUG_PLOT:
        plot_image(ref_img)

    marker_header = (
        'HEADER_MARKER',
        'marker_area',
        'marker_major_axis_length',
        'marker_minor_axis_length',
        'marker_eccentricity'
    )

    marker_data = (
        'MARKER_DATA',
        marker_area,
        major_axis_length,
        minor_axis_length,
        eccentricity
    )
    # Store into global measurements
    if not 'size_marker' in outputs.measurements:
        outputs.measurements['size_marker'] = {}
    outputs.measurements['size_marker']['marker_area'] = marker_area
    outputs.measurements['size_marker']['marker_major_axis_length'] = major_axis_length
    outputs.measurements['size_marker']['marker_minor_axis_length'] = minor_axis_length
    outputs.measurements['size_marker']['marker_eccentricity'] = eccentricity

    # Store images
    outputs.images.append(analysis_image)

    return marker_header, marker_data, analysis_image
