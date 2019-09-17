# for plotting:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for roc curves and auc score:
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score





# function for plotting the ROC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

y_true = matched_true_pred['class']
y_score = matched_true_pred['class_indices']
# #ROC curve
# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
# from matplotlib import pyplot
# from predict_cnn import *
#
# auc = roc_auc_score(images, probs)
# print('AUC: %.3f' % auc)
# # calculate roc curve
# fpr, tpr, thresholds = roc_curve(predict_cnn.images, predict_cnn.pred_pcnt)
# # plot no skill
# pyplot.plot([0, 1], [0, 1], linestyle='--')
# # plot the roc curve for the model
# pyplot.plot(fpr, tpr, marker='.')
# # show the plot
# pyplot.show()
