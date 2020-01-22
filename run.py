import numpy as np
import matplotlib.pyplot as plt

from run_helper import read_sup_dataset, preprocess_set, split_data, plot_roc_curve
from batch_generator import BatchGenerator
from train import train_deep
from test import test
from config import batch_params, train_params, model_config
from models.sentiment_model import SentimentModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def main():
    tweet6k_path = 'dataset/twitter_6K'
    tweet20k_path = 'dataset/twitter_20K'

    # Supervised data
    X_1, y_1 = read_sup_dataset(tweet20k_path)
    X_2, y_2 = read_sup_dataset(tweet6k_path)

    # preprocess data-
    X, y, int2word, word2int = preprocess_set(X_1+X_2, y_1+y_2)

    X_20k, y_20k = X[:len(X_1)], y[:len(y_1)]
    X_6k, y_6k = X[len(X_1):], y[len(y_1):]

    X_6k, X_test, y_6k, y_test = train_test_split(X_6k, y_6k,
                                                  test_size=0.25,
                                                  stratify=y_6k,
                                                  random_state=42)
    X_large = np.concatenate((X_20k, X_6k), axis=0)
    y_large = np.concatenate((y_20k, y_6k), axis=0)

    data_set = [[X_6k, y_6k], [X_large, y_large]]

    figure = plt.figure('roc')
    for set_id, (X, y) in enumerate(data_set):
        if set_id == 0:
            fig_name = 'Kısıtlı veri kümesi'
            train_params['eval_every'] = 30
            ha = 'left'
        else:
            fig_name = 'Çoğaltılmış veri kümesi'
            train_params['eval_every'] = 200
            ha = 'right'

        print('Training for data length: {}'.format(len(y)))
        # split data
        data_dict = split_data(X, y)
        batch_gen = BatchGenerator(data_dict,
                                   set_names=['train', 'validation'],
                                   **batch_params)

        model = SentimentModel(model_config['LSTM'], int2word)
        # train
        train_deep(model, batch_gen, **train_params)

        # test
        log_prob = test(model, X_test, y_test)

        pred = [np.round(prob) for prob in log_prob]

        plot_roc_curve(figure, y_test, log_prob, fig_name=fig_name, ha=ha)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        print('\nTest Accuracy: {}\n'
              'F1 score: {}'.format(acc, f1))

    plt.figure(figure.number)
    plt.savefig('results/final_roc.png', bbox_inches="tight", dpi=100)
    plt.show()


if __name__ == '__main__':
    main()
