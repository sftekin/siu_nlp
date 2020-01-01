import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from preprocessing import Preprocess
from train import train_model


def read_sup_dataset(path):
    labels = ['positive', 'negative', 'notr']
    x = []
    y = []
    for label in labels:
        data_path = os.path.join(path, label + '.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.rstrip()
                x.append(line)
                y.append(label)
    return x, y


def plot_roc_curve(model, data):
    X_train, X_test, y_train, y_test = data
    y_test = np.array(y_test)
    y_score = model.decision_function(X_test)

    fpr = {}
    tpr = {}
    thresholds = {}
    roc_auc = {}

    for idx, label in enumerate(['positive', 'negative', 'notr']):
        y_labels = np.array(y_test != label, dtype=np.int)
        fpr[label], tpr[label], thresholds[label] = roc_curve(y_labels, y_score[:, idx])
        roc_auc[label] = auc(fpr[label], tpr[label])

    plt.figure()
    for label in ['positive', 'negative', 'notr']:
        plt.plot(fpr[label], tpr[label], lw=2,
                 label='{} ROC curve (area = {:.2f})'.format(label, roc_auc[label]))

        optimal_idx = np.argmax(tpr[label] - fpr[label])
        thr_x, thr_y = fpr[label][optimal_idx], tpr[label][optimal_idx]
        threshold = thresholds[label][optimal_idx]
        plt.plot(thr_x, thr_y, 'mv', lw=2)
        plt.text(thr_x, thr_y, s='{:.2f}'.format(threshold), ha='right', va='bottom', fontsize=14)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def main():
    tweet6k_path = 'dataset/twitter_6K/'
    X, y = read_sup_dataset(tweet6k_path)

    pre_pro = Preprocess()
    X, y = pre_pro.transform(X, y)

    data = train_test_split(X, y, test_size=0.2, stratify=y)

    params = {
        'c_list': [0.1, 2, 5, 10],
        'tol': [1e-4, 1e-5],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'linear_svm',
        'load': True
    }

    model6k = train_model(data, **params)

    plot_roc_curve(model6k, data)




if __name__ == '__main__':
    main()
