import os

from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.arcticapi.visuals import plot_px_distribution

def square_im(im):
    width, height = im.shape
    for x in range(width):
        for y in range(height):
            v = im[x, y].item()
            im[x, y] = v * v

def hist(normalized):
    bins = [0]* 256
    for h in range(normalized.shape[0]):
        for w in range(normalized.shape[1]):
            bins[normalized[h, w].item()] += 1

    idxs = [i for i, e in enumerate(bins) if e != 0]
    fig, ax = plt.subplots()
    plt.bar(np.arange(max(idxs)), bins[0:max(idxs)])
    ax.set_yscale('log')
    plt.show()

def lbp_hist(img):
    h, w = img.shape
    hist_vector = np.zeros((256,))
    flat_image = img.flatten()
    for i in range(len(flat_image)):
        # ignore boundary pixels
        if i % w == 0 or i < w or i >= len(flat_image) - w or (i + 1) % w == 0:
            continue
        one = 1 if flat_image[i - (w + 1)] > flat_image[i] else 0
        two = 1 if flat_image[i - (w)] > flat_image[i] else 0
        three = 1 if flat_image[i - (w - 1)] > flat_image[i] else 0
        four = 1 if flat_image[i + 1] > flat_image[i] else 0
        five = 1 if flat_image[i + (w + 1)] > flat_image[i] else 0
        six = 1 if flat_image[i + (w)] > flat_image[i] else 0
        seven = 1 if flat_image[i + (w - 1)] > flat_image[i] else 0
        eight = 1 if flat_image[i - 1] > flat_image[i] else 0
        binary_arr = np.array([one, two, three, four, five, six, seven, eight])
        val = binary_arr.dot(2 ** np.arange(binary_arr.size)[::-1])
        hist_vector[val] += 1
    idxs = [i for i, e in enumerate(hist_vector) if e != 0]
    fig, ax = plt.subplots()
    plt.bar(np.arange(max(idxs)), hist_vector[0:max(idxs)])
    ax.set_yscale('log')
    plt.show()

from PIL import Image
# for filename in glob.glob('/data/raw_data/TrainingAnimals_ThermalImages_00/*.PNG'): #assuming gif
for filename in glob.glob('/data/training_sets/ThermalImages/train_normalized/*.PNG'): #assuming gif

    im = Image.open(filename)
    im.verify()
    continue

    base = os.path.basename(filename)
    im=Image.open(filename)
    im = np.array(im).astype('uint32')

    minval = im.min()
    maxval = im.max()
    if minval != maxval:
        mi = np.percentile(im, 1)
        ma = np.percentile(im, 100)
        normalized = (im - mi) / (ma - mi)

        normalized = normalized * 255
        normalized[normalized < 0] = 0
        normalized = normalized.astype(np.uint8)
        # lbp_hist(normalized)

        pil_im = Image.fromarray(normalized, 'L')
        # plot_px_distribution(im, pil_im, base, 255)


        # pil_im.save('/data/raw_data/TrainingAnimals_ThermalImages_00_norm/'+base)
        pil_im.save('/data/training_sets/ThermalImages/train_normalized/'+base)
