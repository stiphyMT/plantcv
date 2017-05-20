import cv2
import numpy as np
from plantcv import print_image
## collect cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)

### Find Objects Partially Inside Region of Interest or Cut Objects to Region of Interest
def roi_objects(img,roi_type,roi_contour, roi_hierarchy,object_contour, obj_hierarchy, device, debug=False):
 # img = img to display kept objects
 # roi_type = 'cutto' or 'partial' (for partially inside)
 # roi_contour = contour of roi, output from "View and Ajust ROI" function
 # roi_hierarchy = contour of roi, output from "View and Ajust ROI" function
 # object_contour = contours of objects, output from "Identifying Objects" fuction
 # obj_hierarchy = hierarchy of objects, output from "Identifying Objects" fuction
 # device = device number.  Used to count steps in the pipeline
  device +=1
  if len(np.shape(img))==3:
    ix,iy,iz=np.shape(img)
  else:
    ix,iy=np.shape(img)
    
  size = ix,iy,3
  background = np.zeros(size, dtype=np.uint8)
  ori_img=np.copy(img)
  w_back=background+255
  background1 = np.zeros(size, dtype=np.uint8)
  background2 = np.zeros(size, dtype=np.uint8)

  # Allows user to find all objects that are completely inside or overlapping with ROI
  if roi_type=='partial':
    for c,cnt in enumerate(object_contour):
      length=(len(cnt)-1)
      stack=np.vstack(cnt)
      test=[]
      keep=False
      for i in range(0,length):
        pptest=cv2.pointPolygonTest(roi_contour[0], (stack[i][0],stack[i][1]), False)
        if int(pptest)!=-1:
          keep=True
      if keep==True:
        if obj_hierarchy[0][c][3]>-1:
          cv2.drawContours(w_back,object_contour,c, (255,255,255),-1, lineType=8,hierarchy=obj_hierarchy)
        else:  
          cv2.drawContours(w_back,object_contour,c, (0,0,0),-1, lineType=8,hierarchy=obj_hierarchy)
      else:
        cv2.drawContours(w_back,object_contour,c, (255,255,255),-1, lineType=8,hierarchy=obj_hierarchy)
     
    kept=cv2.cvtColor(w_back, cv2.COLOR_RGB2GRAY )
    kept_obj= cv2.bitwise_not(kept)
    mask=np.copy(kept_obj)
    obj_area=cv2.countNonZero(kept_obj)
    if cv2major > 2 and cv2minor > 0:
        _, kept_cnt,hierarchy=cv2.findContours(kept_obj,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        kept_cnt,hierarchy=cv2.findContours(kept_obj,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ori_img,kept_cnt,-1, (0,255,0),-1, lineType=8,hierarchy=hierarchy)
    cv2.drawContours(ori_img,roi_contour,-1, (255,0,0),5, lineType=8,hierarchy=roi_hierarchy)
  
  # Allows uer to cut objects to the ROI (all objects completely outside ROI will not be kept)
  elif roi_type=='cutto':
    cv2.drawContours(background1,object_contour,-1, (255,255,255),-1, lineType=8,hierarchy=obj_hierarchy)
    roi_points=np.vstack(roi_contour[0])
    cv2.fillPoly(background2,[roi_points], (255,255,255))
    obj_roi=cv2.multiply(background1,background2)
    kept_obj=cv2.cvtColor(obj_roi, cv2.COLOR_RGB2GRAY)
    mask=np.copy(kept_obj)
    obj_area=cv2.countNonZero(kept_obj)
    if cv2major > 2 and cv2minor > 0:
        _, kept_cnt,hierarchy = cv2.findContours(kept_obj,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        kept_cnt,hierarchy = cv2.findContours(kept_obj,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(w_back,kept_cnt,-1, (0,0,0),-1)
    cv2.drawContours(ori_img,kept_cnt,-1, (0,255,0),-1, lineType=8,hierarchy=hierarchy)
    cv2.drawContours(ori_img,roi_contour,-1, (255,0,0),5, lineType=8,hierarchy=roi_hierarchy)
         
  else:
    fatal_error('ROI Type' + str(roi_type) + ' is not "cutto" or "partial"!')
  
  if debug:
    print_image(w_back, (str(device) + '_roi_objects.png'))
    print_image(ori_img, (str(device) + '_obj_on_img.png'))
    print_image(mask, (str(device) + '_roi_mask.png'))
    #print ('Object Area=', obj_area)
  
  return device, kept_cnt, hierarchy, mask, obj_area
