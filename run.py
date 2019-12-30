import os
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from embedding.embedding import MeanEmbedding


def read_sup_dataset(path):
    labels = ['positive', 'negative', 'notr']
    x = []
    y = []
    for label in labels:
        data_path = os.path.join(path, label + '.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                words = line.rstrip().split()
                x.append(words)
                y.append(label)
    return x, y


def main():
    tweet6k_path = 'dataset/twitter_6K/'
    embed_path = 'embedding/word2vec.vec'
    X, y = read_sup_dataset(tweet6k_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    sentiment_analysis = Pipeline([
        ('word2vec', MeanEmbedding(embed_path)),
        ('SVM', SVC(kernel='linear'))
    ])

    sentiment_analysis.fit(X_train, y_train)

if __name__ == '__main__':
    main()