import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
import skimage
from src.arcticapi import ArcticApi
from src.arcticapi.config import load_config
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from skimage import data, exposure
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog

from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
import os

class Dataset():
    def __init__(self, config="new_data_ir"):
        self.cfg = load_config(config)
        self.api = ArcticApi(self.cfg)
        # self.image_names = self.api.getImagesWithSeals(True)
        self.image_names = self.api.ir_images.keys()
        self.images = []
        self.flat_data = []
        self.target = []

        # Test/train data
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def load(self, path, a = None, type="none"):
        im = skimage.io.imread(path)
        # self.regional_max(im)
        #if resize is not None:
        # return resize(im, (100, 128), anti_aliasing=True, mode='reflect').flatten()
        minval = im.min()
        maxval = im.max()
        if minval != maxval:
            mi = np.percentile(im, 1)
            ma = np.percentile(im, 90)
            # mi = np.min(im)
            # ma = np.max(im)
            std = np.std(im)
            normalized = (im - mi) / (ma - mi + 0.0)
            normalized = normalized**2

            # self.hog_my(normalized)
            imgplot = plt.imshow(normalized, vmin=normalized.min(), vmax=normalized.max(), cmap='gray')
            plt.title(type)
            plt.show()
            normalized = normalized * 1024
            normalized[normalized < 0] = 0

            bin_counts, bin_edges = np.histogram(normalized, 1024)
            hist_vector = bin_counts / sum(bin_counts + 0.0)

            return hist_vector

    def regional_max(self, image):
        image = gaussian_filter(image, 1)

        h = 0.4
        seed = image - h
        mask = image

        dilated = reconstruction(seed, mask, method='dilation')
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                            ncols=3,
                                            figsize=(8, 2.5),
                                            sharex=True,
                                            sharey=True)

        ax0.imshow(image, cmap='gray')
        ax0.set_title('original image')
        ax0.axis('off')

        ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
        ax1.set_title('dilated')
        ax1.axis('off')

        ax2.imshow(image - dilated, cmap='gray')
        ax2.set_title('image - dilated')
        ax2.axis('off')

        fig.tight_layout()
        plt.show()

    def hog_my(self, image):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
                            cells_per_block=(1,1), visualize=True, multichannel=False)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    def generate(self, out_file, num_seal=500, num_neg=500, dimensions=(64, 64)):
        # LOAD SEAL IMAGES
        for idx, image_name in enumerate(self.image_names):
            if idx > num_seal:
                break
            if not image_name in self.api.ir_images:
                print(image_name)
                continue
            aerial_image = self.api.ir_images[image_name]
            if not aerial_image.load_image():
                print(image_name)
                continue
            if idx % 100 == 0:
                print("Loading seals: %d%%" % int(100 * idx/len(self.image_names)))
            img = self.load(aerial_image.path, type="hs")

            with open(out_file, 'a') as file:
                vals = []
                classIdx = 0
                for pxl_idx, val in enumerate(img):
                    vals.append("%d:%.40f" % (pxl_idx, val))
                line = "%d %s" % (classIdx, ' '.join(vals))
                file.write(line + '\n')

        ## LOAD NEGATIVE BACKGROUND IMAGES
        for root, dirs, files in os.walk("/data/raw_data/TrainingBackground_ThermalImages_00"):
            for idx, file in enumerate(files):
                if idx > num_neg:
                    break
                if file.endswith(".PNG"):
                    if idx % 100 == 0:
                        print("Loading background: %d%%" % int(100 * idx / len(files)))
                    img_resized = self.load(os.path.join(root, file), type="bg")
                    with open(out_file, 'a') as file:
                        vals = []
                        classIdx = 1
                        for pxl_idx, val in enumerate(img_resized):
                            vals.append("%d:%.40f" % (pxl_idx, val))
                        line = "%d %s" % (classIdx, ' '.join(vals))
                        file.write(line + '\n')


def main():
    dataset = Dataset()
    file_name = 'norm_1024bin_normsquared.txt'
    dataset.generate('/data/training_sets/thundersvm/' + file_name,
                     num_seal=10, num_neg=10, dimensions=(64, 80))


    bashCommand = "cat %s | shuf -o %s \n head -n 5000 %s > %s \n tail -n +5000 %s > %s \n rm %s %s" %\
                  ('/data/training_sets/thundersvm/' + file_name,
                   '/data/training_sets/thundersvm/shuf_' + file_name,
                   '/data/training_sets/thundersvm/shuf_' + file_name,
                   '/data/training_sets/thundersvm/test_' + file_name,
                   '/data/training_sets/thundersvm/shuf_' + file_name,
                   '/data/training_sets/thundersvm/train_' + file_name,
                   '/data/training_sets/thundersvm/' + file_name,
                   '/data/training_sets/thundersvm/shuf_' + file_name)
    print bashCommand


if __name__ == "__main__":
    main()
