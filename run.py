import os

from itertools import product
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from embedding.embedding import MeanEmbedding
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


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
    X, y = read_sup_dataset(tweet6k_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    word2vec = MeanEmbedding()

    c_list = [2, 5, 10]
    g_list = [0.5, 1, 2, 4, 5]
    cv = 3

    best_score = 0
    best_params = []
    for c, gamma in product(c_list, g_list):
        clf = SVC(kernel='rbf', C=c, gamma=gamma)

        pipe = Pipeline([
            ('word2vec', word2vec),
            ('clf', clf)
        ])

        # pipe.fit(X_train, y_train)
        cv_score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
        print('C:{}, gamma:{}, cv_score:{}'.format(c, gamma, cv_score))
        cv_score = sum(cv_score) / cv
        if best_score < cv_score:
            best_score = cv_score
            best_params = [c, gamma]

    print('Training finished best params = C:{}, gamma:{}'.format(*best_params))

    clf = SVC(kernel='rbf', C=best_params[0], gamma=best_params[1])
    pipe = Pipeline([
        ('word2vec', word2vec),
        ('clf', clf)
    ])
    pipe.fit(X_train, y_train)
    print("Best parameter Score (CV score=%0.3f):" % pipe.score(X_test, y_test))


if __name__ == '__main__':
    main()
