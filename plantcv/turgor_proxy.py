#!/usr/bin/env python

# Script to generate an image with metrics displayed on image
import sys, traceback
import cv2
import numpy as np
import argparse
import string
import math
from . import print_image
from . import fatal_error
#opencv2 version control
(  cv2major, cv2minor, _) = cv2.__version__.split('.')
(cv2major, cv2minor) = int(major), int(minor)

def turgor_proxy(points_r, centroid_r, bline_r, device, debug=False):
  ## For each point in contour, get a point before (pre) and after (post) the point of interest
  ## The win argument specifies the pre and post point distances
  ## points_r = a set of rescaled points (basically the output of the acute_vertex fxn after the scale_features fxn)
  ## centroid_r = a tuple that contains the rescaled centroid coordinates
  ## bline_r = a tuple that contains the rescaled boundary line - centroid coordinates
  ## device = a count variable
  ## debug = no output supported currently
  #scaled_img = np.zeros((1500,1500,3), np.uint8)
  #plotter = np.array(points_r)
  #plotter = plotter * 1000
  #for i in plotter:
  #  x,y = i.ravel()
  #  cv2.circle(scaled_img,(int(x) + 250, int(y) + 250),15,(255,255,255),-1)
  #cv2.circle(scaled_img,(int(cmx_scaled * 1000) + 250, int(cmy_scaled * 1000) + 250),25,(0,0,255), -1)
  #cv2.circle(scaled_img,(int(blx_scaled * 1000) + 250, int(bly_scaled * 1000) + 250),25,(0,255,0), -1)
  
  device += 1
  vert_dist_c = []
  hori_dist_c = []
  euc_dist_c = []
  angles_c = []
  cx, cy = centroid_r
  ## Do this for centroid
  for pt in points_r:
    ## Get coordinates from point
    x, y = pt
    ## Get vertical distance and append to list
    v = y - cy
    #print "Here is the centroid vertical distance: " + str(v)
    vert_dist_c.append(v)
    #cv2.line(scaled_img, (int((x*1000)+250), int((cy*1000)+250)), (int((x*1000)+250), int((y*1000)+250)), (0,0,255), 5)
    ## Get horizontal distance and append to list
    h = abs(x - cx)
    #print "Here is the centroid horizotnal distance: " + str(h)
    hori_dist_c.append(h)
    e = np.sqrt((cx-x)*(cx-x)+(cy-y)*(cy-y))
    #print "Here is the centroid euclidian distance: " + str(h)
    euc_dist_c.append(e)
    #cv2.line(scaled_img, (int((cx*1000)+250), int((cy*1000)+250)), (int((x*1000)+250), int((y*1000)+250)), (0,165,255), 5)
    #a = (h*h + v*v - e*e)/(2*h*v)
    a = (h*h + e*e - v*v)/(2*h*e)
    if a > 1:              #If float excedes 1 prevent arcos error and force to equal 1
      a = 1
    elif a < -1:           #If float excedes -1 prevent arcos error and force to equal -1
      a = -1      
    ang = abs(math.degrees(math.acos(a)))
    if v < 0:
      ang = ang * -1
    #print "Here is the centroid angle: " + str(ang)
    angles_c.append(ang)
  vert_ave_c = np.mean(vert_dist_c)
  hori_ave_c = np.mean(hori_dist_c)
  euc_ave_c = np.mean(euc_dist_c)
  ang_ave_c = np.mean(angles_c)
  
  vert_dist_b = []
  hori_dist_b = []
  euc_dist_b = []
  angles_b = []
  bx, by = bline_r
  ## Do this for baseline
  for pt in points_r:
    ## Get coordinates from point
    x, y = pt
    ## Get vertical distance and append to list
    v = y - by
    #print "Here is the baseline vertical distance: " + str(v)
    vert_dist_b.append(v)
    #cv2.line(scaled_img, (int((x*1000)+250), int((by*1000)+250)), (int((x*1000)+250), int((y*1000)+250)), (255,255,102), 5)
    ## Get horizontal distance and append to list
    h = abs(x - bx)
    #print "Here is the baseline horizotnal distance: " + str(h)
    hori_dist_b.append(h)
    e = np.sqrt((bx-x)*(bx-x)+(by-y)*(by-y))
    #print "Here is the baseline euclidian distance: " + str(h)
    euc_dist_b.append(e)
    #cv2.line(scaled_img, (int((bx*1000)+250), int((by*1000)+250)), (int((x*1000)+250), int((y*1000)+250)), (255,178,102), 5)
    #a = (h*h + v*v - e*e)/(2*h*v)
    a = (h*h + e*e - v*v)/(2*h*e)
    if a > 1:              #If float excedes 1 prevent arcos error and force to equal 1
      a = 1
    elif a < -1:           #If float excedes -1 prevent arcos error and force to equal -1
      a = -1      
    ang = abs(math.degrees(math.acos(a)))
    if v < 0:
      ang = ang * -1
    #print "Here is the baseline angle: " + str(ang)
    angles_b.append(ang)
  vert_ave_b = np.mean(vert_dist_b)
  hori_ave_b = np.mean(hori_dist_b)
  euc_ave_b = np.mean(euc_dist_b)
  ang_ave_b = np.mean(angles_b)
  #cv2.line(scaled_img, (int(2), int((cy*1000)+250)), (int(1498), int((cy*1000)+250)), (0,215,255), 5)
  #cv2.line(scaled_img, (int(2), int((by*1000)+250)), (int(1498), int((by*1000)+250)), (255,0,0), 5)
  #cv2.circle(scaled_img,(int(cx * 1000) + 250, int(cy * 1000) + 250),25,(0,215,255), -1)
  #cv2.circle(scaled_img,(int(bx * 1000) + 250, int(by * 1000) + 250),25,(255,0,0), -1)
  #flipped_scaled = cv2.flip(scaled_img, 0)
  #cv2.imwrite('centroid_dist.png', flipped_scaled)
  return device, vert_ave_c, hori_ave_c, euc_ave_c, ang_ave_c, vert_ave_b, hori_ave_b, euc_ave_b, ang_ave_b