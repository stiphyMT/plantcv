## Segment a Skeleton 

Turn a skeletonized image into separate pieces. 

**plantcv.morphology.segment_skeleton**(*skel_img, mask=None*)

**returns** Segmented image, segment objects, segment object hierarchies

- **Parameters:**
    - skel_img - Skeleton image (output from [plantcv.morphology.skeletonize](skeletonize.md))
    - mask - Binary mask for debugging (optional). If provided, debug image will be overlaid on the mask.
- **Context:**
    - Breaks skeleton into segments

**Reference Images**

![Screenshot](img/documentation_images/segment_skeleton/skeleton_image.jpg)

![Screenshot](img/documentation_images/segment_skeleton/mask_image.jpg)

```python

from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file), 
# or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "print"

# Adjust line thickness with the global line thickness parameter (default = 5),
# and provide binary mask of the plant for debugging. NOTE: the objects and
# hierarchies returned will be exactly the same but the debugging image (segmented_img)
# will look different.
pcv.params.line_thickness = 3 

segmented_img, obj, hier = pcv.morphology.segment_skeleton(skel_img=skeleton)

segmented_img2, obj, hier = pcv.morphology.segment_skeleton(skel_img=skeleton, 
                                                           mask=plant_mask)

```

*Segmented Image without Mask*

![Screenshot](img/documentation_images/segment_skeleton/segmented_img.jpg)

*Segmented Image with Mask*

![Screenshot](img/documentation_images/segment_skeleton/segmented_img_mask.jpg)
