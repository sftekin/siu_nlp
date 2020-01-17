import numpy as np
import pickle
from run_helper import read_sup_dataset, preprocess_set, split_data
from batch_generator import BatchGenerator
from train import train
from test import test
from config import batch_params, train_params, model_config
from models.sentiment_model import SentimentModel
from sklearn.model_selection import train_test_split


def main():
    tweet6k_path = 'dataset/twitter_6K'
    tweet20k_path = 'dataset/twitter_20K'

    # Supervised data
    X_1, y_1 = read_sup_dataset(tweet20k_path)
    X_2, y_2 = read_sup_dataset(tweet6k_path)

    # preprocess data
    X, y, int2word, word2int = preprocess_set(X_1+X_2, y_1+y_2)

    X_20k, y_20k = X[:len(X_1)], y[:len(y_1)]
    X_6k, y_6k = X[len(X_1):], y[len(y_1):]

    X_6k, X_test, y_6k, y_test = train_test_split(X_6k, y_6k,
                                                  test_size=0.2,
                                                  stratify=y_6k,
                                                  random_state=42)
    X = np.concatenate((X_20k, X_6k), axis=0)
    y = np.concatenate((y_20k, y_6k), axis=0)

    # split data
    data_dict = split_data(X, y)

    batch_gen = BatchGenerator(data_dict,
                               set_names=['train', 'validation'],
                               **batch_params)

    model = SentimentModel(model_config, int2word)

    train(model, batch_gen, **train_params)

    # test
    model_file = open('bilstm_model.pkl', 'rb')
    model = pickle.load(model_file)
    data_dict = {'test': (X_test, y_test)}
    batch_gen = BatchGenerator(data_dict,
                               set_names=['test'],
                               **batch_params)
    log_prob, acc = test(model, batch_gen)

    print('Test Accuracy: {}'.format(acc))


if __name__ == '__main__':
    main()
