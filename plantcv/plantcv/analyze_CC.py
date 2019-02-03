import os as os
import numpy as np
import cv2 
import sys, traceback
import argparse
import string
import warnings
from copy import deepcopy
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import fatal_error
from plantcv.plantcv import params
from plantcv.plantcv import naive_bayes_classifier
from plantcv.plantcv import PCVconstants as pcvc



def _pseudocoloured_image_( rgb_img, masks, ccnames, colours = None):
    """
    Pseudocolor image.

    Inputs:
    rgb_img         = input image
    masks           = binary mask images
    ccnames         = names of the colourclasses
    colours         = colors for false colour image
    filename        = input image filename
    analysis_images = list of analysis image filenames

    Returns:
    analysis_images = list of analysis image filenames

    :param histogram: list
    :param bins: int
    :param img: numpy array
    :param mask: numpy array
    :param background: str
    :param channel: str
    :param filename: str
    :param analysis_images: list
    :return analysis_images: list
    """
    """ 
    Function to create a false coloured image of x colourclasses
    input:
        masks: a list of masks for each colour class
        colours: a list of colours for each colour class (optional), if colour are not specified colours will be automatically chosen
        
    output:
        np.arry bgr image file
    """
    # remove first (ie header)
    masks, ccnames = masks[1:], ccnames[1:]
    x, y = masks[0].shape
    colr_number = len( masks)

    # there should be twice as many names due to Absolute and Relative calculations
    if colr_number * 2 != len( ccnames):
        raise Exception( 'Number of masks is not equal to colour class names.')
    elif colours is not None and colr_number < len( colours):
        colours = None
        warnings.warn("Warning: Not enough colours passed, selecting colours automatically.")
    elif colours is None:
        if colr_number == 4:
            # standard set of DarkGreen, Green, Yellow and Brown
            colours = [[0,128,0],[0,255,0],[0,255,255],[0,64,128]]
        else:
            # create a list of colours for output
            h = np.arange( 0, 255, np.ceil( 255 / colr_number), np.uint8)
            v = np.repeat( 200, colr_number).astype( np.uint8)
            s = np.repeat( 200, colr_number).astype( np.uint8)
            colours = cv2.merge([ h, s, v])
            colours = cv2.cvtColor( colours, cv2.COLOR_HSV2BGR)
    else:
        pass
        
    # create a copy of the rgb_image as greyscale
    l = cv2.cvtColor( rgb_img, cv2.COLOR_BGR2LAB)[:,:,0]
    false_coloured_image = cv2.merge( (l,l,l))
    
    # apply different colour for each channel mask to greyscale image
    font = cv2.FONT_HERSHEY_DUPLEX
    for ncc, tg in enumerate( zip(masks, colours)):
        # create an colour image, same size as original, with a solid colour
        temp = np.zeros(( x, y, 3), np.uint8)
        temp[:] = tg[1]
        # copy coloured pixels through mask to original image (now greyscale)
        false_coloured_image[ np.where( tg[0] == 255)] = temp[ np.where( tg[0] == 255)]
        cv2.putText( false_coloured_image, ccnames[2 * ncc][:-8], ( 20, 100 + (ncc * 50)), font, 2, tg[1], 1, cv2.LINE_AA)
        
        
    
    
    return [false_coloured_image] 


def analyze_CC( ori_img, plant_mask = None, pdfs = None, colours = None, filename = False):
    """
    Analyze the color class areas of an image object

    Inputs:
    rgb_img          = RGB image data
    plant_mask        = Binary mask made from selected contours
    pdfs             = filename for a probability density function (pdfs) file created with naive_bayes_multiclass
    colours          = a list of colours for the the false colour image, a list will be created if none specified
    filename         = False or image name. If defined print image

    Returns:
    hist_header      = color histogram data table headers
    hist_data        = color histogram data table values
    analysis_images  = list of output images

    :param rgb_img: numpy.ndarray
    :param mask: numpy.ndarray
    :param pdfs: str
    :param colours: list
    :param filename: str
    :return colourclass_header: list
    :return colourclass_data: list
    :return analysis_images: list
    """
    params.device += 1
    
    # if there is no plant_mask create a blank mask (all white) 
    if plant_mask is None:
        x, y = ori_img.shape[:2]
        plant_mask = np.zeros( (x, y), np.uint8)
        plant_mask[:] = 255
        warnings.warn("Warning: No mask specified, using a blank mask.")
        
    
    # analyze the image with the probability density function (pdfs) multiple channel file
    # a mask for each channel is returned in a dictionary from the whole image
    cc_masks = naive_bayes_classifier( ori_img, pdfs)


    # mask the channel masks against the plant mask and copy them to a list
    colourclass_masks = [ cv2.bitwise_and( cc_masks[i], plant_mask) for i in cc_masks.keys()]
    
    # find area of whole plant mask, for use in relative area calculations
    m = cv2.moments( plant_mask, binaryImage = True)
    total_area = m['m00']
    
    # calculate the white pixels (area) for each channel + original plant mask area
    colourclass_data = []
    for i in colourclass_masks:
        m = cv2.moments( i[plant_mask == 255], binaryImage = True)
        area = m['m00']
        colourclass_data.extend( [area, area / total_area])
    #[np.sum( i[plant_mask == 255]) for i in colourclass_masks ]
    
    # copy the names of the masks from the dictionary to a list
    colourclass_header = []
    for i in cc_masks.keys():
        colourclass_header.extend( [i + ' Absolute', i + ' Relative'])
        
    # insert name for the colourclass masks list at position 0
    colourclass_masks.insert( 0, 'COLOURCLASSES_MASKS')
    # insert name for the colourclass channel names list at position 0
    colourclass_header.insert( 0, 'HEADER_COLOURCLASSES')
    # insert name for the colourclass calculated area list at position 0
    colourclass_data.insert( 0, 'COLOURCLASSES_DATA')

    # create a false colour image of the colourclasses
    out_images = _pseudocoloured_image_( ori_img, colourclass_masks, colourclass_header, colours)
    
    if filename:
        # Output images with boundary line, above/below bound area
        out_file = os.path.splitext( filename)[0] + '_colourclasses_' + str( len(cc_masks)) + '.jpg'
        print_image( out_images[0], out_file)
        analysis_images = [['IMAGE', 'colour_class', out_file]]
        

    if params.debug == pcvc.DEBUG_PRINT:
        print_image( out_img, os.path.join( params.debug_outdir, str( params.device) + '_' + str( len(cc_masks)) +'_cc.jpg'))
    elif params.debug == pcvc.DEBUG_PLOT:
        if len( np.shape( out_img)) == 3:
            plot_image( out_img)
        else:
            plot_image( out_img, cmap = pcvc.COLOUR_MAP_GREY)

    return colourclass_header, colourclass_data, analysis_images

