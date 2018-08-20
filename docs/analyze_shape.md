## Analyze Shape Characteristics of Object

Shape analysis outputs numeric properties for an input object (contour or grouped contours), works best on grouped contours.
 
**analyze_object**(*img, imgname, obj, mask, device, debug=None, filename=False*)

**returns** device, shape data headers, shape data, image with shape data

- **Parameters:**
    - img - image object (most likely the original), color(RGB)
    - imgname - name of image
    - obj - single or grouped contour object
    - device - Counter for image processing steps
    - debug - None, "print", or "plot". Print = save to file, Plot = print to screen. Default = None 
    - filename - False or image name. If defined print image
- **Context:**
    - Used to output shape characteristics of an image, including height, object area, convex hull, convex hull area, perimeter, extent x, extent y, longest axis, centroid x coordinate, centroid y coordinate, in bounds QC (if object touches edge of image, image is flagged). 
- **Example use:**
    - [Use In VIS Tutorial](../vis_tutorial.html)
    - [Use In NIR Tutorial](nir_tutorial.md)
    - [Use In PSII Tutorial](psII_tutorial.md) 
    
**Original image**

![Screenshot](img/documentation_images/analyze_shape/original_image.jpg)

```python
from plantcv import plantcv as pcv

# Characterize object shapes
    
device, shape_header, shape_data, shape_img = pcv.analyze_object(img, imgname, objects, mask, device, debug="print", /home/malia/setaria_shape_img.png)
```

**Image with identified objects**

![Screenshot](img/documentation_images/analyze_shape/objects_on_image.jpg)

**Image with shape characteristics**

![Screenshot](img/documentation_images/analyze_shape/shapes_on_image.jpg)
