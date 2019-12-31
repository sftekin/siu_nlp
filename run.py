import os

from sklearn.model_selection import train_test_split
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


def main():
    tweet6k_path = 'dataset/twitter_6K/'
    X, y = read_sup_dataset(tweet6k_path)

    pre_pro = Preprocess()
    X, y = pre_pro.transform(X, y)

    data = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_test, y_train, y_test = data


    params = {
        'c_list': [0.1, 2, 5, 10],
        'tol': [1e-4, 1e-5],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'linear_svm',
        'load': True
    }

    model6k = train_model(data, **params)
    confidences = model6k.decision_function(X_test)
    print(confidences)


if __name__ == '__main__':
    main()
