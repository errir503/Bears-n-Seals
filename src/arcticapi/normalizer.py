"""Functions for transforming 16-bit thermal images into 8-bits"""
import cv2
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
def normalize_ir_global(camerapos, filePath, colorJet, bit_8=True):
    img = PILImage.open(filePath)
    if img is None:
        return None
    img = np.array(img)
    bottom, top = camera_bounds(camerapos, img.shape[0])
    normalized = lin_normalize_image(img, bit_8, bottom, top)
    if colorJet:
        normalized = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_HSV)
    return normalized


def normalize_ir_local(filePath, colorJet, bit_8=True):
    img = PILImage.open(filePath)
    if img is None:
        return None
    img = np.array(img)
    normalized = lin_normalize_image(img, bit_8)
    if colorJet:
        normalized = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_HSV)
    return normalized


def norm(fileIR, colorJet, percent=0.01):
    img = cv2.imread(fileIR, cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    img = np.floor((img - np.percentile(img, percent)) / (
            np.percentile(img, 100 - percent) - np.percentile(img, percent)) * 256)

    if colorJet:
        img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_HSV)
    return img

def norm2(fileIR, colorJet, percent=0.01):
    img = cv2.imread(fileIR)
    normalizedImg = np.zeros(img.shape)
    normalizedImg = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    if colorJet:
        normalizedImg = cv2.applyColorMap(normalizedImg.astype(np.uint8), cv2.COLORMAP_HSV)
    return normalizedImg

