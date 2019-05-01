# Analyzes an object and outputs numeric properties

import os
import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import params
from plantcv.plantcv import outputs
from plantcv.plantcv import PCVconstants as pcvc


def analyze_object( img, obj, mask):
    """Outputs numeric properties for an input object (contour or grouped contours).

    Inputs:
    img             = RGB or grayscale image data for plotting
    obj             = single or grouped contour object
    mask            = Binary image to use as mask for moments analysis

    Returns:
    shape_header    = shape data table headers
    shape_data      = shape data table values
    analysis_images = list of output images

    :param img: numpy.ndarray
    :param obj: list
    :param mask: numpy.ndarray
    :return shape_header: list
    :return shape_data: list
    :return analysis_images: list
    """

    params.device += 1

    # Valid objects can only be analyzed if they have >= 5 vertices
    if len(obj) < 5:
        return None, None, None

    ori_img = np.copy( img)
    # Convert grayscale images to color
    if len( np.shape(ori_img)) == 2:
        ori_img = cv2.cvtColor( ori_img, cv2.COLOR_GRAY2BGR)

    if len( np.shape( img)) == 3:
        ix, iy, iz = np.shape( img)
    else:
        ix, iy = np.shape( img)
    size = ix, iy, 3
    size1 = ix, iy
    background = np.zeros( size, dtype = np.uint8)
    background1 = np.zeros( size1, dtype = np.uint8)
    background2 = np.zeros( size1, dtype = np.uint8)

    # Check is object is touching image boundaries (QC)
    frame_background = np.zeros( size1, dtype = np.uint8)
    frame = frame_background + 1
    if pcvc.CV2MAJOR > 2 and pcvc.CV2MINOR > 0:
        _, frame_contour, frame_heirarchy = cv2.findContours( frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        frame_contour, frame_heirarchy = cv2.findContours( frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ptest = []
    vobj = np.vstack( obj)
    for i, c in enumerate( vobj):
        xy = tuple( c)
        pptest = cv2.pointPolygonTest( frame_contour[0], xy, measureDist = False)
        ptest.append( pptest)
    in_bounds = all( c == 1 for c in ptest)

    # Convex Hull
    hull = cv2.convexHull( obj)
    hull_vertices = len( hull)
    # Moments
    #  m = cv2.moments(obj)
    m = cv2.moments( mask, binaryImage = True)
    # Properties
    # Area
    area = m['m00']

    if area:
        # Convex Hull area
        hull_area = cv2.contourArea( hull)
        # Solidity
        solidity = 1
        if int( hull_area) != 0:
            solidity = area / hull_area
        # Perimeter
        perimeter = cv2.arcLength( obj, closed = True)
        # compactness as the "isoperimetric quotient"
        compactness = ( 4 * np.pi * area) / ( perimeter ** 2)
        # x and y position (bottom left?) and extent x (width) and extent y (height)
        x, y, width, height = cv2.boundingRect( obj)
        # Centroid (center of mass x, center of mass y)
        cmx, cmy = ( m['m10'] / m['m00'], m['m01'] / m['m00'])
        # Ellipse
        center, axes, angle = cv2.fitEllipse( obj)
        major_axis = np.argmax( axes)
        minor_axis = 1 - major_axis
        major_axis_length = axes[major_axis]
        minor_axis_length = axes[minor_axis]
        eccentricity = np.sqrt( 1 - ( axes[minor_axis] / axes[major_axis]) ** 2)

        # Longest Axis: line through center of mass and point on the convex hull that is furthest away
        cv2.circle( background, ( int( cmx), int( cmy)), 4, (255, 255, 255), -1)
        center_p = cv2.cvtColor( background, cv2.COLOR_BGR2GRAY)
        ret, centerp_binary = cv2.threshold( center_p, 0, 255, cv2.THRESH_BINARY)
        if pcvc.CV2MAJOR > 2 and pcvc.CV2MINOR > 0:
            _, centerpoint, cpoint_h = cv2.findContours( centerp_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            centerpoint, cpoint_h = cv2.findContours( centerp_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        dist = []
        vhull = np.vstack(hull)

        for i, c in enumerate( vhull):
            xy = tuple( c)
            pptest = cv2.pointPolygonTest( centerpoint[0], xy, measureDist=True)
            dist.append( pptest)

        abs_dist = np.absolute( dist)
        max_i = np.argmax( abs_dist)

        caliper_max_x, caliper_max_y = list( tuple( vhull[max_i]))
        caliper_mid_x, caliper_mid_y = [ int(cmx), int(cmy)]

        xdiff = float( caliper_max_x - caliper_mid_x)
        ydiff = float( caliper_max_y - caliper_mid_y)

        # Set default values
        slope = 1

        if xdiff != 0:
            slope = ( float( ydiff / xdiff))
        b_line = caliper_mid_y - ( slope * caliper_mid_x)

        if slope != 0:
            xintercept = int( -b_line / slope)
            xintercept1 = int(( ix - b_line) / slope)
            if 0 <= xintercept <= iy and 0 <= xintercept1 <= iy:
                cv2.line(background1, (xintercept1, ix), (xintercept, 0), (255), params.line_thickness)
            elif xintercept < 0 or xintercept > iy or xintercept1 < 0 or xintercept1 > iy:
                # Used a random number generator to test if either of these cases were possible but neither is possible
                # if xintercept < 0 and 0 <= xintercept1 <= iy:
                #     yintercept = int(b_line)
                #     cv2.line(background1, (0, yintercept), (xintercept1, ix), (255), 5)
                # elif xintercept > iy and 0 <= xintercept1 <= iy:
                #     yintercept1 = int((slope * iy) + b_line)
                #     cv2.line(background1, (iy, yintercept1), (xintercept1, ix), (255), 5)
                # elif 0 <= xintercept <= iy and xintercept1 < 0:
                #     yintercept = int(b_line)
                #     cv2.line(background1, (0, yintercept), (xintercept, 0), (255), 5)
                # elif 0 <= xintercept <= iy and xintercept1 > iy:
                #     yintercept1 = int((slope * iy) + b_line)
                #     cv2.line(background1, (iy, yintercept1), (xintercept, 0), (255), 5)
                # else:
                yintercept = int(b_line)
                yintercept1 = int((slope * iy) + b_line)
                cv2.line(background1, (0, yintercept), (iy, yintercept1), (255), 5)
        else:
            cv2.line(background1, (iy, caliper_mid_y), (0, caliper_mid_y), (255), params.line_thickness)

        ret1, line_binary = cv2.threshold( background1, 0, 255, cv2.THRESH_BINARY)
        # print_image(line_binary,(str(device)+'_caliperfit.png'))

        cv2.drawContours( background2, [hull], -1, (255), -1)
        ret2, hullp_binary = cv2.threshold( background2, 0, 255, cv2.THRESH_BINARY)
        # print_image(hullp_binary,(str(device)+'_hull.png'))

        caliper = cv2.multiply( line_binary, hullp_binary)
        # print_image(caliper,(str(device)+'_caliperlength.png'))

        caliper_y, caliper_x = np.array( caliper.nonzero())
        caliper_matrix = np.vstack(( caliper_x, caliper_y))
        caliper_transpose = np.transpose( caliper_matrix)
        caliper_length = len( caliper_transpose)

        caliper_transpose1 = np.lexsort(( caliper_y, caliper_x))
        caliper_transpose2 = [( caliper_x[i], caliper_y[i]) for i in caliper_transpose1]
        caliper_transpose = np.array( caliper_transpose2)

    # else:
    #  hull_area, solidity, perimeter, width, height, cmx, cmy = 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND'

    # Store Shape Data
    shape_header = [
        'HEADER_SHAPES',
        'area',
        'hull-area',
        'solidity',
        'compactness',
        'perimeter',
        'width',
        'height',
        'longest_axis',
        'center-of-mass-x',
        'center-of-mass-y',
        'hull_vertices',
        'in_bounds',
        'ellipse_center_x',
        'ellipse_center_y',
        'ellipse_major_axis',
        'ellipse_minor_axis',
        'ellipse_angle',
        'ellipse_eccentricity'
    ]

    shape_data = [
        'SHAPES_DATA',
        area,
        hull_area,
        solidity,
        compactness,
        perimeter,
        width,
        height,
        caliper_length,
        cmx,
        cmy,
        hull_vertices,
        in_bounds,
        center[0],
        center[1],
        major_axis_length,
        minor_axis_length,
        angle,
        eccentricity
    ]

    analysis_images = []

    # Draw properties
    if area:
        cv2.drawContours(ori_img, obj, -1, (255, 0, 0), params.line_thickness)
        cv2.drawContours(ori_img, [hull], -1, (0, 0, 255), params.line_thickness)
        cv2.line(ori_img, (x, y), (x + width, y), (0, 0, 255), params.line_thickness)
        cv2.line(ori_img, (int(cmx), y), (int(cmx), y + height), (0, 0, 255), params.line_thickness)
        cv2.line(ori_img, (tuple(caliper_transpose[caliper_length - 1])), (tuple(caliper_transpose[0])), (0, 0, 255),
                 params.line_thickness)
        cv2.circle(ori_img, (int(cmx), int(cmy)), 10, (0, 0, 255), params.line_thickness)
        # Output images with convex hull, extent x and y
        # out_file = os.path.splitext(filename)[0] + '_shapes.jpg'
        # out_file1 = os.path.splitext(filename)[0] + '_mask.jpg'

        # print_image(ori_img, out_file)
        analysis_images.append(ori_img)

        # print_image(mask, out_file1)
        analysis_images.append(mask)

    else:
        pass

    # Store into global measurements
    if not "shapes" in outputs.measurements:
        outputs.measurements["shapes"] = {}
    outputs.measurements["shapes"]["area"] = area
    outputs.measurements["shapes"]["hull-area"] = hull_area
    outputs.measurements["shapes"]["solidity"] = solidity
    outputs.measurements["shapes"]["perimeter"] = perimeter
    outputs.measurements["shapes"]["width"] = width
    outputs.measurements["shapes"]["height"] = height
    outputs.measurements["shapes"]["longest_axis"] = caliper_length
    outputs.measurements["shapes"]["center-of-mass-x"] = cmx
    outputs.measurements["shapes"]["center-of-mass-y"] = cmy
    outputs.measurements["shapes"]["hull_vertices"] = hull_vertices
    outputs.measurements["shapes"]["in_bounds"] = in_bounds
    outputs.measurements["shapes"]["ellipse_center_x"] = center[0]
    outputs.measurements["shapes"]["ellipse_center_y"] = center[1]
    outputs.measurements["shapes"]["ellipse_major_axis"] = major_axis_length
    outputs.measurements["shapes"]["ellipse_minor_axis"] = minor_axis_length
    outputs.measurements["shapes"]["ellipse_angle"] = angle
    outputs.measurements["shapes"]["ellipse_eccentricity"] = eccentricity

    if params.debug is not None:
        cv2.drawContours(ori_img, obj, -1, (255, 0, 0), params.line_thickness)
        cv2.drawContours(ori_img, [hull], -1, (0, 0, 255), params.line_thickness)
        cv2.line(ori_img, (x, y), (x + width, y), (0, 0, 255), params.line_thickness)
        cv2.line(ori_img, (int(cmx), y), (int(cmx), y + height), (0, 0, 255), params.line_thickness)
        cv2.circle(ori_img, (int(cmx), int(cmy)), 10, (0, 0, 255), params.line_thickness)
        cv2.line(ori_img, (tuple(caliper_transpose[caliper_length - 1])), (tuple(caliper_transpose[0])), (0, 0, 255),
                 params.line_thickness)
        if params.debug == 'print':
            print_image(ori_img, os.path.join(params.debug_outdir, str(params.device) + '_shapes.jpg'))
        elif params.debug == 'plot':
            if len(np.shape(img)) == 3:
                plot_image(ori_img)
            else:
                plot_image( ori_img, cmap = pcvc.COLOUR_MAP_GREY)

    # Store images
    outputs.images.append(analysis_images)
    return shape_header, shape_data, analysis_images
