import cv2

COLOR2GRAY_HSV_H = 'h'
COLOR2GRAY_HSV_S = 's'
COLOR2GRAY_HSV_V = 'v'

COLOR2GRAY_LAB_L = 'l'
COLOR2GRAY_LAB_A = 'a'
COLOR2GRAY_LAB_B = 'b'

COLOR2GRAY_RGB_B = 'b'
COLOR2GRAY_RGB_G = 'g'
COLOR2GRAY_RGB_R = 'r'

THRESHOLD_OBJ_DARK = 'dark'
THRESHOLD_OBJ_LIGHT = 'light'

APPLY_MASK_WHITE = 'white'
APPLY_MASK_BLACK = 'black'

DEFINE_ROI_SHAPE_RECT = 'rectangle'
DEFINE_ROI_SHAPE_CIRCLE = 'circle'
DEFINE_ROI_INPUT_BINARY = 'binary'
DEFINE_ROI_INPUT_RGB = 'rgb'

ROI_OBJECTS_TYPE_CUTTO = 'cutto'
ROI_OBJECTS_TYPE_PARTIAL = 'partial'
ROI_OBJECTS_TYPE_MASSC = 'massc'

DEBUG_PRINT = 'print'
DEBUG_PLOT = 'plot'

LINEBLUR_HORI = 0
LINEBLUR_VERT = 1

## collect cv2 version info
try:
    cv2major, cv2minor, _, _ = cv2.__version__.split('.')
except:
    cv2major, cv2minor, _ = cv2.__version__.split('.')
cv2major, cv2minor = int(cv2major), int(cv2minor)