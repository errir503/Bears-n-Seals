"""Functions for transforming 16-bit thermal images into 8-bits"""
import numpy as np
from PIL import Image as PILImage

def norm_matrix(m):
    if m < 0:
        return 0
    if m > 1:
        return 1
    return m

def norm_matrix2(m):
    if m < 0:
        return 0
    if m > 1:
        return 1
    if m > .5:
        return 0
    return m

def lin_normalize_image(image_array, bit_8, bottom=None, top=None):
    """Linear normalization for an image array
    Inputs:
        image_array: np.ndarray, image data to be normalized
        bit_8: boolean, if true outputs 8 bit, otherwise outputs 16 bit
        bottom: float, value to map to 0 in the new array
        top: float, value to map to 2^(bit_depth)-1 in the new array
    Output:
        scaled_image: nd.ndarray, scaled image between 0 and 2^(bit_depth) - 1
    """
    if bottom is None:
        bottom = np.min(image_array)
    if top is None:
        top = np.max(image_array)
    scaled_image = (image_array - bottom + 0.0) / (top - bottom + 0.0)
    scaled_image = np.vectorize(norm_matrix)(scaled_image)

    if bit_8:
        scaled_image = np.floor(scaled_image * 255).astype(np.uint8)  # Map to [0, 2^8 - 1]
    else:
        scaled_image = np.floor(scaled_image * 65535).astype(np.uint16)  # Map to [0, 2^16 - 1]

    return scaled_image

def camera_bounds(camera_pos, num_rows):
    # camera_pos S and default
    bottom = 51000
    top = 57500

    if camera_pos == "P":
        if num_rows == 512:
            bottom = 53500
            top = 56500
        elif num_rows == 480:
            bottom = 50500
            top = 58500
    elif camera_pos == "C":
        bottom = 50500
        top = 58500

    return bottom, top

## Normalize for specific camera
def normalize_ir_global(camerapos, filePath, bit_8 = True):
    img =  np.array(PILImage.open(filePath))
    bottom, top = camera_bounds(camerapos, img.shape[0])
    normalized = lin_normalize_image(img, bit_8, bottom, top)
    return normalized

def normalize_ir_local(camerapos, filePath, bit_8 = True):
    img =  np.array(PILImage.open(filePath))
    normalized = lin_normalize_image(img, bit_8)
    return normalized

def norm(img, percent=0.01):
    return np.floor((img - np.percentile(img, percent)) / (
            np.percentile(img, 100 - percent) - np.percentile(img, percent)) * 256)


    

