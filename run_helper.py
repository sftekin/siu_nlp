import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers.preprocessing import Preprocess
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split


def read_sup_dataset(path, load=True):
    pickle_name = path.split('/')[-1] + '.pkl'
    save_path = os.path.join(path, pickle_name)
    if os.path.isfile(save_path) and load:
        save_file = open(save_path, 'rb')
        x, y = pickle.load(save_file)
        print('{} loaded'.format(pickle_name))
        return x, y
    print('Reading supervised dataset, {}'.format(pickle_name))
    labels = ['negative', 'positive']
    x = []
    y = []
    for label in labels:
        data_path = os.path.join(path, label + '.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.rstrip()
                x.append(line)
                y.append(labels.index(label))
    save_file = open(save_path, 'wb')
    pickle.dump((x, y), save_file)
    return x, y


def preprocess_set(x, y, seq_len=15):
    pre_pro = Preprocess(seq_len)
    x, y = pre_pro.transform(x, y)

    # pad x
    pad_x = []
    for twt in x:
        if len(twt) < seq_len:
            twt += ['<pad>']*(seq_len - len(twt))
        pad_x.append(twt)
    x = pad_x

    # construct vocab
    flat_x = [word for sentence in x for word in sentence]

    # sort the vocab according to occurrences
    counts = {}
    for word in flat_x:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}

    int2word = {ii: word for ii, word in enumerate(counts.keys())}
    word2int = {word: ii for ii, word in int2word.items()}

    # convert x to integers
    x = [[word2int[word] for word in twt] for twt in x]

    # convert to numpy array
    x = np.array(x)
    y = np.array(y)
    return x, y, int2word, word2int


def split_data(X, y, val_ratio=0.1):
    X_train, X_val, \
    y_train, y_val = train_test_split(X, y,
                                      test_size=val_ratio,
                                      stratify=y,
                                      random_state=42)
    data_dict = {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
    }
    return data_dict


def plot_roc_curve(figure, y_test, y_score, fig_name='', ha='right'):
    print('plotting roc curves')
    y_labels = np.array(y_test == 1, dtype=np.int)
    fpr, tpr, threshold = roc_curve(y_labels, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figure.number)
    plt.plot(fpr, tpr, lw=2,
             label='{} AİK Eğrisi (alan = {:.2f})'.format(fig_name, roc_auc))

    optimal_idx = np.argmax(tpr - fpr)
    thr_x, thr_y = fpr[optimal_idx], tpr[optimal_idx]
    threshold = threshold[optimal_idx]
    plt.plot(thr_x, thr_y, 'mv', lw=2)
    plt.text(thr_x, thr_y, s='{:.2f}'.format(threshold), ha=ha, va='bottom', fontsize=10)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Yanlış Pozitif Sıklığı')
    plt.ylabel('Doğru Pozitif Sıklığı')
    plt.title('Alıcı İşletim Karakteristik Eğrisi')
    plt.grid(True)
    plt.legend(loc="lower right")
