## Closing

Filters out dark noise from an image.

**plantcv.closing**(*gray_img, kernel=None*)

**returns** filtered_img

- **Parameters:**
    - gray_img - Grayscale or binary image data
    - kernel - Optional neighborhood, expressed as an array of 1's and 0's. If None, 
    use cross-shaped structuring element.
  - **Context:**
    - Used to reduce image noise, specifically small dark spots (i.e. "pepper").
- **Example use:**
    - See below

```python

from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file), or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "print"

# Apply closing

filtered_img = pcv.closing(gray_img)

```

**Grayscale image**

![Screenshot](img/documentation_images/closing/before_closing.jpg)

**Closing**

![Screenshot](img/documentation_images/closing/after_closing.jpg)
