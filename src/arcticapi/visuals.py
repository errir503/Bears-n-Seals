import sys

import cv2
import matplotlib.pyplot as plt

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

# Print iterations progress
# w/modifications from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r')
    sys.stdout.write('%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()