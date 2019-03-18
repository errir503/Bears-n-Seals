from thundersvmScikit import *
from sklearn.datasets import *
import matplotlib.pyplot as plt

traning_file = "train_64_80_seals.txt"
train_set_path = "/data/training_sets/thundersvm/" + traning_file
x,y = load_svmlight_file(train_set_path)
# clf = NuSVC(verbose=True, gamma=0.001, kernel='rbf', gpu_id=0, nu=1)
clf = SVC(verbose=True, gamma=0.001, C=100, kernel='rbf', gpu_id=0)
clf.fit(x,y)
clf.save_to_file('./model2')

