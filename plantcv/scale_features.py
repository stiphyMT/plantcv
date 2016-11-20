#!/usr/bin/env python

## Function to return feature scaled points
import sys, traceback
import cv2
import numpy as np
import argparse
import string
import math
from . import print_image
from . import plot_image
from . import fatal_error
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(cv2major), int(cv2minor)

def scale_features(obj, mask, points, boundary_line, device, debug=False):
  ## This is a function to transform the coordinates of landmark points onto a common scale (0 - 1.0)
  ## obj = a contour of the plant object (this should be output from the object_composition.py fxn)
  ## mask = this is a binary image. The object should be white and the background should be black
  ## points = the points to scale
  ##
  ## device = a counter variable
  ## img = This is a copy of the original plant image generated using np.copy if debug is true it will be drawn on
  ## debug = True/False. If True, print image
  device += 1
  ## Get the dimensions of the image from the binary thresholded object (mask)
  ix, iy = np.shape(mask)
  x,y,width,height = cv2.boundingRect(obj)
  m = cv2.moments(mask, binaryImage=True)
  cmx,cmy = (m['m10']/m['m00'], m['m01']/m['m00'])
  ## Convert the boundary line position (top of the pot) into a coordinate on the image
  if boundary_line != 'NA':
    line_position=int(ix)-int(boundary_line)
    bly = line_position
  else:
    bly = cmy
  blx = cmx
  ## Maximum and minimum values of the object 
  Ymax = y
  Ymin = y + height
  Xmin = x
  Xmax = x + width
  ## Scale the coordinates of each of the feature locations
  ## Feature scaling X' = (X - Xmin) / (Xmax - Xmin)
  ## Feature scaling Y' = (Y - Ymin) / (Ymax - Ymin)
  rescaled = []
  for p in points:
    xval = float(p[0,0] - Xmin) / float(Xmax - Xmin)
    yval = float(p[0,1] - Ymin) / float(Ymax - Ymin)
    scaled_point = (xval,yval)
    rescaled.append(scaled_point)
  ## Lets rescale the centroid
  cmx_scaled = float(cmx - Xmin) / float(Xmax - Xmin)
  cmy_scaled = float(cmy - Ymin) / float(Ymax - Ymin)
  centroid_scaled = (cmx_scaled, cmy_scaled)
  ## Lets rescale the boundary_line
  blx_scaled = float(blx - Xmin) / float(Xmax - Xmin)
  bly_scaled = float(bly - Ymin) / float(Ymax - Ymin)
  boundary_line_scaled = (blx_scaled, bly_scaled)
  ## If debug is 'True' plot an image of the scaled points on a black background
  if debug == 'print':
    ## Make a decent size blank image
    scaled_img = np.zeros((1500,1500,3), np.uint8)
    plotter = np.array(rescaled)
    ## Multiple the values between 0 - 1.0 by 1000 so you can plot on the black image
    plotter = plotter * 1000
    ## For each of the coordinates plot a circle where the point is (+250 helps center the object in the middle of the blank image)
    for i in plotter:
      x,y = i.ravel()
      cv2.circle(scaled_img,(int(x) + 250, int(y) + 250),15,(255,255,255),-1)
    cv2.circle(scaled_img,(int(cmx_scaled * 1000) + 250, int(cmy_scaled * 1000) + 250),25,(0,0,255), -1)
    cv2.circle(scaled_img,(int(blx_scaled * 1000) + 250, int(bly_scaled * 1000) + 250),25,(0,255,0), -1)
    ## Because the coordinates increase as you go down and to the right on the image you need to flip the object around the x-axis
    flipped_scaled = cv2.flip(scaled_img, 0)
    cv2.imwrite((str(device) + '_feature_scaled.png'), flipped_scaled)
  ## Return the transformed points
  if debug == 'plot':
    ## Make a decent size blank image
    scaled_img = np.zeros((1500,1500,3), np.uint8)
    plotter = np.array(rescaled)
    ## Multiple the values between 0 - 1.0 by 1000 so you can plot on the black image
    plotter = plotter * 1000
    ## For each of the coordinates plot a circle where the point is (+250 helps center the object in the middle of the blank image)
    for i in plotter:
      x,y = i.ravel()
      cv2.circle(scaled_img,(int(x) + 250, int(y) + 250),15,(255,255,255),-1)
    cv2.circle(scaled_img,(int(cmx_scaled * 1000) + 250, int(cmy_scaled * 1000) + 250),25,(0,0,255), -1)
    cv2.circle(scaled_img,(int(blx_scaled * 1000) + 250, int(bly_scaled * 1000) + 250),25,(0,255,0), -1)
    ## Because the coordinates increase as you go down and to the right on the image you need to flip the object around the x-axis
    flipped_scaled = cv2.flip(scaled_img, 0)
    plot_image(flipped_scaled)
  ## Return the transformed points

  return device, rescaled, centroid_scaled, boundary_line_scaled