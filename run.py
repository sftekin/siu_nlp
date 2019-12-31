import os
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from embedding.embedding import MeanEmbedding
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
                x.append(words)
                y.append(label)
    return x, y


def main():
    tweet6k_path = 'dataset/twitter_6K/'
    embed_path = 'embedding'
    X, y = read_sup_dataset(tweet6k_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    word2vec = MeanEmbedding(embed_path)
    clf = SVC(kernel='rbf')

    params = {
        'clf__C': [0.001, 0.01, 1, ],
        'clf__gamma': [2 ** -4, 2 ** -1, 1]
    }

    pipe = Pipeline([
        ('word2vec', word2vec),
        ('clf', clf)
    ])

    search = GridSearchCV(pipe, params, scoring='accuracy', n_jobs=-1, verbose=3)
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % pipe.score(X_test, y_test))


if __name__ == '__main__':
    main()
