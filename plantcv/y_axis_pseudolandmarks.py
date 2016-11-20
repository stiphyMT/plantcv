#!/usr/bin/env python

## Function to scan for pseudolandmarks along the y-axis
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
(cv2major, cv2minor) = int(major), int(minor)

def y_axis_pseudolandmarks(obj, mask, img, device, debug = False):
  ## Divide up object contour into 19 equidistance segments and generate landmarks for each
  ## Obj = a contour of the plant object (this should be output from the object_composition.py fxn)
  ## mask = this is a binary image. The object should be white and the background should be black
  ## device = a counter variable
  ## img = This is a copy of the original plant image generated using np.copy if debug is true it will be drawn on
  ## debug = True/False. If True, print image
  ##### Lets get some landmarks scanning along the y-axis
  device += 1
  x,y,width,height = cv2.boundingRect(obj)
  extent = height
  ## If height is greater than 21 pixels make 20 increments (5% intervals)
  if extent >= 21:
    inc = extent / 21
    # Define variable for max points and min points
    pts_max = []
    pts_min = []
    # Get max and min points for each of the intervals
    for i in range(1,21):
      if (i == 1):
        pt_max = y
        pt_min = y + (inc * i)
      else:
        pt_max = y + (inc * (i-1))
        pt_min = y + (inc * i)
      # Put these in an array
      pts_max.append(pt_max)
      pts_min.append(pt_min)
    # Combine max and min into a set of tuples
    point_range = list(zip(pts_max,pts_min))
    # define some list variables to fill
    row_median=[]
    row_ave=[]
    max_width=[]
    left_points=[]
    right_points=[]
    y_vals=[]
    x_centroids=[]
    y_centroids=[]
    # For each of the 20 intervals
    for pt in point_range:
      # Get the lower and upper bounds  (lower and higher in terms of value; low point is actually towards top of photo, higher is lower of photo)
      low_point, high_point = pt
      # Get all rows within these two points
      rows=[]
      lps=[]
      rps=[]
      # Get a continuous list of the values between the top and the bottom of the interval save as vals
      vals = list(range(low_point, high_point))
      # For each row... get all coordinates from object contour that match row
      for v in vals:
        # Value is all entries that match the row
        value = obj[v == obj[:,0,1]]
        if len(value) > 0:
          # Could potentially be more than two points in all contour in each pixel row
          # Grab largest x coordinate (column)
          largest = value[:,0,0].max()
          # Grab smallest x coordinate (column)
          smallest = value[:,0,0].min()
          # Take the difference between the two (this is how far across the object is on this plane)
          row_width = largest - smallest
          # Append this value to a list
          rows.append(row_width)
          lps.append(smallest)
          rps.append(largest)
        if len(value) == 0:
          row_width = 1
          rows.append(row_width)
          lps.append(1)
          rps.append(1)
      # For each of the points find the median and average width
      row_median.append(np.median(np.array(rows)))
      row_ave.append(np.mean(np.array(rows)))
      max_width.append(np.max(np.array(rows)))
      left_points.append(np.mean(smallest))
      right_points.append(np.mean(largest))
      yval = int((high_point+low_point)/2)
      y_vals.append(yval)
      # Make a copy of the mask; we want to get landmark points from this
      window = np.copy(mask)
      window[:low_point] = 0
      window[high_point:] = 0
      s = cv2.moments(window)
      # Centroid (center of mass x, center of mass y)
      if (largest - smallest > 3):
          smx,smy = (s['m10']/s['m00'], s['m01']/s['m00'])
          x_centroids.append(int(smx))
          y_centroids.append(int(smy))
      else:
        smx = (largest + smallest)/2
        smy = yval
        x_centroids.append(int(smx))
        y_centroids.append(int(smy))
    # Get the indicie of the largest median/average x-axis value (if there is a tie it takes largest index)
    #indice_median = row_median.index(max(row_median))
    #indice_ave = row_ave.index(max(row_ave))
    #median_value = row_median[indice_median]
    #ave_value = row_ave[indice_ave]
    #max_value = max_width[indice_ave]
    left = zip(left_points,y_vals)
    left = np.array(left)
    left.shape = (20,1,2)
    right = list(zip(right_points,y_vals))
    right = np.array(right)
    right.shape = (20,1,2)
    center_h = list(zip(x_centroids, y_centroids))
    center_h = np.array(center_h)
    center_h.shape = (20,1,2)
    if debug == 'plot':
      img2 = np.copy(img)
      for i in left:
        x = i[0,0]
        y = i[0,1]
        cv2.circle(img2,(int(x),int(y)),10,(255,0,0),-1)
      for i in right:
        x = i[0,0]
        y = i[0,1]
        cv2.circle(img2,(int(x),int(y)),10,(255,0,255),-1)
      for i in center_h:
        x = i[0,0]
        y = i[0,1]
        cv2.circle(img2,(int(x),int(y)),10,(0,79,255),-1)
      #print_image(img2, (str(device) + '_y_axis_pseudolandmarks.png'))
      plot_image(img2)
    return device, left, right, center_h
  
  if extent < 21:
    # If the length of the object is less than 20 pixels just make the object a 20 pixel rectangle
    x,y,width,height = cv2.boundingRect(obj)
    y_coords = list(range(y, y+20))
    l_points = [x] * 20
    left = list(zip(l_points, y_coords))
    left = np.array(left)
    left.shape = (20,1,2)
    r_points = [x + width] * 20
    right = list(zip(r_points, y_coords))
    right = np.array(right)
    right.shape = (20,1,2)
    m = cv2.moments(mask, binaryImage=True)
    # Centroid (center of mass x, center of mass y)
    cmx,cmy = (m['m10']/m['m00'], m['m01']/m['m00'])
    c_points = [cmx] * 20
    center_h = list(zip(c_points, y_coords))
    center_h = np.array(center_h)
    center_h.shape = (20,1,2)
    if debug == 'plot':
      img2 = np.copy(img)
      for i in left:
        x = i[0,0]
        y = i[0,1]
        cv2.circle(img2,(int(x),int(y)),10,(255,0,0),-1)
      for i in right:
        x = i[0,0]
        y = i[0,1]
        cv2.circle(img2,(int(x),int(y)),10,(255,0,255),-1)
      for i in center_h:
        x = i[0,0]
        y = i[0,1]
        cv2.circle(img2,(int(x),int(y)),10,(0,79,255),-1)
      #print_image(img2, (str(device) + '_y_axis_pseudolandmarks.png'))
      plot_image(img2)
    return device, left, right, center_h