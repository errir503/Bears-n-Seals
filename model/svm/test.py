from thundersvmScikit import *
from sklearn.datasets import *
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
test_file = "test_hist_512_norm.txt"
test_set_path = "/data/training_sets/thundersvm/" + test_file
clf = SVC(verbose=True, C=100, gamma=0.1, kernel="rbf")
clf.load_from_file("./train_64_80_seals.txt.model")

x2,y2=load_svmlight_file(test_set_path)
y_predict=clf.predict(x2)
# score=clf.score(x2,y2)
accuracy = accuracy_score(y2, y_predict)
recall = precision_recall_fscore_support(y2, y_predict)
true_positives = [0, 0]
false_negatives = [0, 0]
total_counts = [0, 0]
for idx, truth in enumerate(y2):
    pred = int(y_predict[idx])
    truth = int(truth)
    total_counts[truth] += 1
    if truth == pred:
        true_positives[truth] += 1
    else:
        false_negatives[truth] += 1
print("Total Seals: %d, Total Background: %d" % (total_counts[0], total_counts[1]))
print("Seals Correct: %d, Background Correct: %d" % (true_positives[0], true_positives[1]))
print("Seals Wrong: %d, Background Wrong: %d" % (false_negatives[0], false_negatives[1]))
print ("SEAL - precision = %0.2f, recall = %0.2f, F1 = %0.2f" % \
           (recall[0][0], recall[1][0], recall[2][0]))
print ("Background - precision = %0.2f, recall = %0.2f, F1 = %0.2f" % \
           (recall[0][1], recall[1][1], recall[2][1]))
print "Accuracy ", accuracy