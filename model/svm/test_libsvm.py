import pickle

from sklearn import svm
test_file = "test_64_80_seals.txt"
test_set_path = "/data/training_sets/thundersvm/" + test_file
clf = svm.SVC()
s = pickle.dumps(clf)
clf.load_from_file("./train_64_80_seals.txt.model")

pass
clf.fit(X, y)