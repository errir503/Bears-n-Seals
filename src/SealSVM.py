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

from skimage.io import imread
from skimage.transform import resize
import os

class SealSVM():
    def __init__(self, config="new_data_ir"):
        self.cfg = load_config(config)
        self.api = ArcticApi(self.cfg)
        self.image_names = self.api.getImagesWithSeals(True)
        self.images = []
        self.flat_data = []
        self.target = []

        # Test/train data
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def load_images(self, num_seal=500, num_neg=500, dimensions=(64, 64)):
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
            img = skimage.io.imread(aerial_image.path)
            img_resized = resize(img, dimensions, anti_aliasing=True, mode='reflect')
            self.flat_data.append(img_resized.flatten())
            self.images.append(img_resized)
            aerial_image.free()
            self.target.append(0)

        ## LOAD NEGATIVE BACKGROUND IMAGES
        for root, dirs, files in os.walk("/data/raw_data/TrainingBackground_ThermalImages_00"):
            for idx, file in enumerate(files):
                if idx > num_neg:
                    break
                if file.endswith(".PNG"):
                    if idx % 100 == 0:
                        print("Loading background: %d%%" % int(100 * idx / len(files)))
                    img = skimage.io.imread(os.path.join(root, file))
                    img_resized = resize(img, dimensions, anti_aliasing=True, mode='reflect')
                    self.flat_data.append(img_resized.flatten())
                    self.images.append(img_resized)
                    self.target.append(1)

    def split(self, test_size = 0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.flat_data, self.target, test_size=test_size)

    def train(self, kernel='linear'):

        C_range = np.array([1, 10, 100, 1000])
        gamma_range = np.array([0.001, 0.0001])
        param_grid = [
            {'C': C_range, 'kernel': ['linear']},
            {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']},
        ]
        svc = svm.SVC()
        clf = GridSearchCV(svc, param_grid)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        print("Classification report for - \n{}:\n{}\n".format(
            clf, metrics.classification_report(self.y_test, y_pred)))

    def plot(self):
        pca = PCA(n_components=2).fit(self.X_train)
        pca_2d = pca.transform(self.X_train)
        import pylab as pl
        c1, c2 = None, None
        for i in range(0, pca_2d.shape[0]):
            if self.y_train[i] == 0:
                c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
            elif self.y_train[i] == 1:
                c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        pl.legend([c1, c2], ['None', 'Seal'])
        pl.show()

    def save_to_file(self):
        # save train file
        with open('/data/training_sets/thundersvm/train_128_160_all.txt', 'w') as file:
            for idx, x in enumerate(self.X_train):
                vals = []
                classIdx = self.y_train[idx]
                for pxl_idx, val in enumerate(x):
                    vals.append("%d:%f" % (pxl_idx, val))
                line = "%d %s" % (classIdx, ' '.join(vals))
                file.write(line + '\n')

        with open('/data/training_sets/thundersvm/test_128_160_all.txt', 'w') as file:
            for idx, x in enumerate(self.X_test):
                vals = []
                classIdx = self.y_test[idx]
                for pxl_idx, val in enumerate(x):
                    vals.append("%d:%f" % (pxl_idx, val))
                line = "%d %s" % (classIdx, ' '.join(vals))
                file.write(line + '\n')
        # save test file



def main():
    svm = SealSVM()
    svm.load_images(num_seal=100000, num_neg=100000, dimensions=(128,160))
    svm.split(test_size=0.5)
    svm.save_to_file()
    # svm.train()


if __name__ == "__main__":
    main()
