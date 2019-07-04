# Author: Daniel Guggenheim
# Company: unisante, iumsp
# Date: 03/2019
# ===================================================
"""
Printing metrics and other utility functions for machine learning.
"""

import matplotlib.pyplot as plt
import itertools

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    :param y_true: np.array: the correct label for y
    :param y_pred: np.array: the predicted label for y
    :param classes: Array of labels for the values of y.
    :param normalize: Normalization can be applied by setting `normalize=True`.
    :param title: Title of plot
    :param cmap: The color of the plot.
    """
    if classes is None:
        classes = ['Negative', 'Positive']

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def print_metrics(y_true, y_pred):
    """
    Print some interesting metrics for a vector of true label and a vector of predicted value.

    :param y_true: np.array: the correct label for y
    :param y_pred: np.array: the predicted label for y
    """
    positive, negative, true_positive, false_positive, true_negative, false_negative = [0] * 6

    for i, (y_true_, y_pred_) in enumerate(zip(y_true, y_pred)):
        if y_pred_ == 1:
            positive += 1
            if y_true_ == 1:
                true_positive += 1
            else:
                false_positive += 1
        else:
            negative += 1
            if y_true_ == 0:
                true_negative += 1
            else:
                false_negative += 1

    print(f'Predicted positive: {positive}\nPredicted negative: {negative}\nTrue positive: {true_positive}')
    print(f'False positive: {false_positive}\nTrue negative: {true_negative}\nFalse negative: {false_negative}')
    print(f'Sensitivity=recall=vp/(vp+fn): {true_positive / max(1, true_positive + false_negative):.2f}')
    print(f'Specificity=vn/(vn+fp): {true_negative / max(1, true_negative + false_positive):.2f}')
    print(f'VPP=vp/(vp+fp): {true_positive / max(1, true_positive + false_positive):.2f}')
    print(f'VPN=vn/(vn+fn): {true_negative / max(1, true_negative + false_negative):.2f}')


def print_all_metrics(y_true, y_pred):
    print_metrics(y_true, y_pred)
    # print('\n' + '-' * 30 + '\n')
    # plot_confusion_matrix(y_true, y_pred, ['Negative', 'Positive'], normalize=False, title='Confusion matrix', )
    print('\n' + '-' * 30 + '\n')
    print(metrics.classification_report(y_true, y_pred))
