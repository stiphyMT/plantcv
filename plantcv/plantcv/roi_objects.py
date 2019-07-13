import cv2
import numpy as np
import os
import re
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error
from plantcv.plantcv import params
from plantcv.plantcv import PCVconstants as pcvc


def roi_objects( img, roi_type, roi_contour, roi_hierarchy, object_contour, obj_hierarchy):
    """Find objects partially inside a region of interest or cut objects to the ROI.

    Inputs:
    img            = RGB or grayscale image data for plotting
    roi_type       = 'cutto' or 'partial' (for partially inside), 'largest' (keep only the largest contour) or 'massc' (for center of mass inside)
    roi_contour    = contour of roi, output from "View and Adjust ROI" function
    roi_hierarchy  = contour of roi, output from "View and Adjust ROI" function
    object_contour = contours of objects, output from "find_objects" function
    obj_hierarchy  = hierarchy of objects, output from "find_objects" function

    Returns:
    kept_cnt       = kept contours
    hierarchy      = contour hierarchy list
    mask           = mask image
    obj_area       = total object pixel area

    :param img: numpy.ndarray
    :param roi_type: str
    :param roi_contour: list
    :param roi_hierarchy: numpy.ndarray
    :param object_contour: list
    :param obj_hierarchy: numpy.ndarray
    :return kept_cnt: list
    :return hierarchy: numpy.ndarray
    :return mask: numpy.ndarray
    :return obj_area: int
    """

    params.device += 1
    # Create an empty grayscale (black) image the same dimensions as the input image
    mask = np.zeros( np.shape( img)[:2], dtype = np.uint8)

    # Make a copy of the input image for plotting
    ori_img = np.copy( img)
    # If the reference image is grayscale convert it to color
    if len(np.shape( ori_img)) == 2:
        ori_img = cv2.cvtColor( ori_img, cv2.COLOR_GRAY2BGR)

    # Allows user to find all objects that are completely inside or overlapping with ROI
    if roi_type.upper() == pcvc.ROI_OBJECTS_TYPE_PARTIAL or roi_type.upper() == pcvc.ROI_OBJECTS_TYPE_LARGEST:
        # Filter contours outside of the region of interest
        for c, cnt in enumerate( object_contour):
            length = ( len( cnt) - 1)
            stack = np.vstack( cnt)
            keep = False

            # Test if the contours are within the ROI
            for r, rcnt in enumerate( roi_contour):
                if keep:
                    break
                for i in range(0, length):
                    pptest = cv2.pointPolygonTest( rcnt, ( stack[i][0], stack[i][1]), False)
                    if int(pptest) != -1:
                        keep = True
            if keep:
                # Color the "gap contours" white
                if obj_hierarchy[0][c][3] > -1:
                    cv2.drawContours( mask, object_contour, c, (0), -1, lineType = 8, hierarchy = obj_hierarchy)
                else:
                    # Color the plant contour parts black
                    cv2.drawContours( mask, object_contour, c, (255), -1, lineType = 8, hierarchy = obj_hierarchy)
            else:
                # If the contour isn't overlapping with the ROI, color it white
                cv2.drawContours(mask, object_contour, c, (0), -1, lineType=8, hierarchy = obj_hierarchy)

        # Find the kept contours and area
        kept_cnt, kept_hierarchy = cv2.findContours(np.copy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        obj_area = cv2.countNonZero(mask)

        # Find the largest contour if roi_type is set to 'largest'
        if roi_type.upper() == pcvc.ROI_OBJECTS_TYPE_LARGEST:
            # Print warning statement about this feature
            print("Warning: roi_type = 'largest' will only return the largest contour and its immediate children. Other "
                  "subcontours will be dropped.")
            # Find the index of the largest contour in the list of contours
            largest_area = 0
            for c, cnt in enumerate( kept_cnt):
                area = cv2.contourArea( cnt)
                if area > largest_area:
                    largest_area = area
                    index = c

            # Store the largest contour as a list
            largest_cnt = [kept_cnt[index]]

            # Store the hierarchy of the largest contour into a list
            largest_hierarchy = [kept_hierarchy[0][index]]

            # Iterate through contours to find children of the largest contour
            for i, khi in enumerate(kept_hierarchy[0]):
                if khi[3] == index:  # is the parent equal to the largest contour?
                    largest_hierarchy.append( khi)
                    largest_cnt.append( kept_cnt[i])

            # Make the kept hierarchies into an array so that cv2 can use it
            largest_hierarchy = np.array( [largest_hierarchy])

            # Overwrite mask so it only has the largest contour
            mask = np.zeros(np.shape( img)[:2], dtype = np.uint8)
            for i, cnt in enumerate( largest_cnt):
                # print( cnt)
                if i == 0:
                    color = (255)
                else:
                    color = (0)
                    # print(i)
                cv2.drawContours(mask, largest_cnt, i, color, -1, lineType = 8, hierarchy = largest_hierarchy, maxLevel = 0)

            # Refind contours and hierarchy from new mask so they are easier to work with downstream
            kept_cnt, kept_hierarchy = cv2.findContours( np.copy( mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

            # Compute object area
            obj_area = cv2.countNonZero( mask)

        cv2.drawContours( ori_img, kept_cnt, -1, ( 0, 255, 0), -1, lineType = 8, hierarchy = kept_hierarchy)
        cv2.drawContours( ori_img, roi_contour, -1, ( 255, 0, 0), params.line_thickness, lineType = 8,
                         hierarchy = roi_hierarchy)
    # Allows user to cut objects to the ROI (all objects completely outside ROI will not be kept)
    elif roi_type.upper() == pcvc.ROI_OBJECTS_TYPE_CUTTO:
        background1 = np.zeros(np.shape(img)[:2], dtype=np.uint8)
        background2 = np.zeros(np.shape(img)[:2], dtype=np.uint8)
        cv2.drawContours( background1, object_contour, -1, ( 255), -1, lineType = 8, hierarchy = obj_hierarchy)
        # roi_points = np.vstack( roi_contour[0])
        # cv2.fillPoly( background2, [roi_points], ( 255))
        cv2.drawContours( background2, roi_contour, -1, (255), cv2.FILLED, lineType = 8, hierarchy = roi_hierarchy)
        mask = cv2.multiply(background1, background2)
        obj_area = cv2.countNonZero( mask)
        kept_cnt, kept_hierarchy = cv2.findContours(np.copy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours( ori_img, kept_cnt, -1, ( 0, 255, 0), -1, lineType = 8, hierarchy = kept_hierarchy)
        cv2.drawContours( ori_img, roi_contour, -1, ( 255, 0, 0), params.line_thickness, lineType = 8,
                         hierarchy=roi_hierarchy)

    elif 'PCT' in roi_type.upper():
        roi_type = re.sub('PCT$', '', roi_type)
        try:
            roi_type = int(roi_type)
        except ValueError:
            roi_type = 50
        finally:
            if  0 < roi_type < 100:
                pct_cutoff = roi_type / 100
            else:
                pct_cutoff = 0.5
        background1 = np.zeros(np.shape(img)[:2], dtype=np.uint8)
        background2 = np.zeros(np.shape(img)[:2], dtype=np.uint8)
        roi_points = np.vstack( roi_contour[0])
        cv2.drawContours( background2, roi_contour, -1, (255), cv2.FILLED, lineType = 8, hierarchy = roi_hierarchy)
        for c, cnt in enumerate( object_contour):
            back_obj = np.copy( background1)
            # Test if the contours are within the ROI
            cv2.drawContours( back_obj, object_contour, c, (255), cv2.FILLED, lineType = 8, hierarchy = obj_hierarchy)
            comb = cv2.multiply( back_obj, background2)
            try:
                frac = cv2.countNonZero( comb) / cv2.countNonZero( back_obj)
            except ZeroDivisionError:
                frac = 0
            if frac > pct_cutoff:
                # Color the "gap contours" white
                if obj_hierarchy[0][c][3] > -1:
                    cv2.drawContours( mask, object_contour, c, (0), -1, lineType = 8, hierarchy = obj_hierarchy)
                else:
                    # Color the plant contour parts black
                    cv2.drawContours( mask, object_contour, c, (255), -1, lineType = 8, hierarchy = obj_hierarchy)
            else:
                # If the contour isn't overlapping with the ROI, color it black
                cv2.drawContours(mask, object_contour, c, (0), -1, lineType=8, hierarchy = obj_hierarchy)
        # Find the kept contours and area
        kept_cnt, kept_hierarchy = cv2.findContours(np.copy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        obj_area = cv2.countNonZero(mask)



    # Cuts off objects that do not have their center of mass inside the ROI
    elif roi_type == pcvc.ROI_OBJECTS_TYPE_MASSC:
        # p_cnt = 
        for c, cnt in enumerate( object_contour):
            # stack = np.vstack( cnt)
            # test = []
            # keep = False
            # calculate objects Moments
            obj_moment = cv2.moments( cnt)
            # calculate object/contour center of mass position from Moments dictionary
            try:
                cx, cy = int( obj_moment['m10'] / obj_moment['m00']), int( obj_moment['m01'] / obj_moment['m00'])
            except ZeroDivisionError:
                pptest = -1
            else:
                # determine whether the center of mass in in or outside the roi
                for r, rcnt in enumerate( roi_contour):
                    pptest = cv2.pointPolygonTest( rcnt, ( cx, cy), False)
                    if pptest > -1:
                        break
            # pptest = -1 if it is outside the roi
            if int( pptest) != -1:
                if obj_hierarchy[ 0][ c][ 3] > -1:
                    cv2.drawContours( mask, object_contour, c, ( 0, 0, 0), -1, lineType = 8, hierarchy = obj_hierarchy)
                else:  
                    cv2.drawContours( mask, object_contour, c, ( 255, 255, 255), -1, lineType = 8, hierarchy = obj_hierarchy)
            else:
                cv2.drawContours( mask, object_contour, c, ( 0, 0, 0), -1, lineType = 8, hierarchy = obj_hierarchy)
     
        # kept = cv2.cvtColor( mask, cv2.COLOR_RGB2GRAY )
        # kept_obj = cv2.bitwise_not( kept)
        # mask = np.copy( kept_obj)
        obj_area = cv2.countNonZero( mask)
        if pcvc.CV2MAJOR >= 3 and pcvc.CV2MINOR >= 1:
            _, kept_cnt, kept_hierarchy = cv2.findContours( mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            kept_cnt, kept_hierarchy = cv2.findContours( mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours( ori_img, kept_cnt, -1, ( 0, 255, 0), -1, lineType = 8, hierarchy = kept_hierarchy)
        cv2.drawContours( ori_img, roi_contour, -1, ( 255, 0, 0), 5, lineType = 8, hierarchy = roi_hierarchy)

    else:
        fatal_error( 'ROI Type {0} is not "cutto", "largest", "partial", "XXpct" or "massc"!'.format( roi_type))

    if params.debug == pcvc.DEBUG_PRINT:
        print_image( ori_img, os.path.join( params.debug_outdir, str( params.device) + '_obj_on_img.png'))
        print_image( mask, os.path.join( params.debug_outdir, str( params.device) + '_roi_mask.png'))
    elif params.debug == pcvc.DEBUG_PLOT:
        plot_image( ori_img)
        plot_image( mask, cmap = pcvc.COLOR_MAP_GRAY)

    return kept_cnt, kept_hierarchy, mask, obj_area
