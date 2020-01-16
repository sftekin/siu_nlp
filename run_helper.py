import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, f1_score


def read_sup_dataset(path, pre_pro, load=True):
    save_path = os.path.join(path, 'tweet_20k.pkl')
    if os.path.isfile(save_path) and load:
        save_file = open(save_path, 'rb')
        x, y, features = pickle.load(save_file)
        print('tweet_20k.pkl loaded')
        return (x, y), features
    print('Reading supervised dataset')
    labels = ['positive', 'negative']
    x = []
    y = []
    for label in labels:
        data_path = os.path.join(path, label + '.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.rstrip()
                x.append(line)
                y.append(labels.index(label))

    (x, y), features = pre_pro.transform(x, y)
    save_file = open(save_path, 'wb')
    pickle.dump((x, y, features), save_file)
    return (x, y), features


def read_unsup_dataset(path, pre_pro, sample_size=1e5, load=True):
    save_path = os.path.join(path, 'tweet_100k.pkl')
    if os.path.isfile(save_path) and load:
        save_file = open(save_path, 'rb')
        x = pickle.load(save_file)
        print('tweet_100k.pkl loaded')
        return x

    print('Reading UNsupervised dataset')
    x = []
    data_path = os.path.join(path, 'tweets.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.rstrip()
            x.append(line)
    x = np.array(x)
    x = x[np.random.permutation(len(x))]
    x = x[:int(sample_size)].tolist()

    x = pre_pro.transform(x)
    save_file = open(save_path, 'wb')
    pickle.dump(x, save_file)
    return x


def plot_roc_curve(confidence_fun, X_test, y_test, fig_name=''):
    print('plotting roc curves')
    save_path = os.path.join('results', fig_name + '.png')
    y_test = np.array(y_test)
    y_score = confidence_fun(X_test)

    fpr = {}
    tpr = {}
    thresholds = {}
    roc_auc = {}

    for idx, label in enumerate(['positive', 'negative', 'notr']):
        y_labels = np.array(y_test == idx, dtype=np.int)
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
        plt.text(thr_x, thr_y, s='{:.2f}'.format(threshold), ha='right', va='bottom', fontsize=10)
        thresholds[label] = threshold  # store only the bests

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    return thresholds


def self_label(confidence_fun, word2vec, data, threshold):
    x = []
    y = []
    # labels = ['positive', 'negative', 'notr']
    y_scores = confidence_fun(word2vec.transform(data))
    pred_class = np.argmax(y_scores, axis=1)

    for data_idx, (score, pred) in enumerate(zip(y_scores, pred_class)):
        # thr = thresholds[labels[pred]]
        if score[pred] > threshold:
            x.append(data[data_idx])
            y.append(pred)
    print(len(x))
    return x, y


def compare_models(models, X_test, y_test, model_names):
    for idx, model_name in enumerate(model_names):
        y_pred = models[idx].predict(X_test, y_test)
        macro = f1_score(y_test, y_pred, average='macro')
        micro = f1_score(y_test, y_pred, average='micro')
        weighted = f1_score(y_test, y_pred, average='weighted')
        print('{} --> f1_macro: {}, '
              'f1_micro: {}, f1_weighted: {}'.format(model_name, macro, micro, weighted))
