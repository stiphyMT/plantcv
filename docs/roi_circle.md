## Create a circular Region of Interest (ROI)

**plantcv.roi.circle**(*img, x, y, r*)

**returns** roi_contour, roi_hierarchy

- **Parameters:**
    - img - An RGB or grayscale image to plot the ROI on in debug mode.
    - x - The x-coordinate of the center of the circle.
    - y - The y-coordinate of the center of the circle.
    - r - The radius of the circle.
- **Context:**
    - Used to define a region of interest in the image.

**Reference Image**

![Screenshot](img/documentation_images/circle/original_image.jpg)

```python

from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file), 
# or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "print"

roi_contour, roi_hierarchy = pcv.roi.circle(img=rgb_img, x=200, y=225, r=75)

```

![Screenshot](img/documentation_images/circle/image_with_roi.jpg)
