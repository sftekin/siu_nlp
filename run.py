import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def read_sup_dataset(path):
    labels = ['positive', 'negative', 'notr']
    x = []
    y = []
    for label in labels:
        data_path = os.path.join(path, label + '.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                words = line.rstrip().split()
                x.append([word.lower() for word in words])
                y.append(label)
    return x, y


def main():
    tweet6k_path = 'dataset/twitter_6K/'
    X, y = read_sup_dataset(tweet6k_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    print()


if __name__ == '__main__':
    main()