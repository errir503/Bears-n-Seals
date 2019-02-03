import sys

import cv2
import matplotlib.pyplot as plt

from model.HotSpot import ColorsList


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def show_ir(hsm, colorJet = False):
    for hs in hsm.hotspots:
        if not hs.ir.load_image(colorJet):
            continue
        cv2.imshow('norm', hs.ir.image[0])
        cv2.imshow('anydepth', hs.ir.image[1])
        cv2.imshow('normglobal', hs.ir.image[2])
        cv2.imshow('normlocal', hs.ir.image[3])
        cv2.waitKey(0)

def norm_matrix(m):
    if m < 0:
        return 0
    if m > 1:
        return 1
    return m

# draw a bbox and center point of bbox on image given yolo labels
def drawBBoxYolo(img, x, y, w, h, label):
    color = ColorsList[label]
    (imh, imw, imc) = img.shape
    x = int(x * imw)
    y = int(y * imh)
    w = int(w * imw)
    h = int(h * imh)
    cv2.circle(img, (x, y), 5, color, 2)
    cv2.rectangle(img, (x - w / 2, y - h / 2),
                  (x + w / 2, y + h / 2),
                  color, 2)  # draw rect

def pltIm(img):
    imgplot = plt.imshow(img)
    plt.show()

def plot_px_distribution(imgpre, imgpost, title, bins):
    # bottom = np.percentile(imgpre, 1)
    # top = np.percentile(imgpre, 100)
    # scaled_image = (imgpre - bottom + 0.0) / (top - bottom + 0.0)
    # scaled_image = np.vectorize(norm_matrix)(scaled_image)

    plt.title("Pre Norm")
    plt.hist(imgpre, fc='red',rwidth=1, bins='auto')
    plt.show()

    # bottom = np.percentile(imgpost, 1)
    # top = np.percentile(imgpost, 100)
    # scaled_image = (imgpost - bottom + 0.0) / (top - bottom + 0.0)
    # scaled_image = np.vectorize(norm_matrix)(scaled_image)

    plt.title("Post Norm")
    plt.hist(imgpost, fc='red',rwidth=1, bins='auto')
    plt.show()



def plot_16bit_gray(img, cmap="gray"):
    """
    Plot a 16-bit grayscale image
    @params:
        img   - Image to plot
        cmap   - Matplotlib color map (Str)
    """
    plt.imshow(img, vmin=0, vmax=65535, cmap=cmap)
    plt.show()

def print_loading_bar(pct):
    sys.stdout.write("\r|%-73s| %3d%%" % ('#' * int(pct * .73), pct))
