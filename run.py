"""
"""
from run_helper import plot_roc_curve, read_unsup_dataset, read_sup_dataset
from sklearn.model_selection import train_test_split
from preprocessing import Preprocess
from train import train_model


def main():
    tweet100k_path = 'dataset/twitter_100K'
    tweet6k_path = 'dataset/twitter_6K'

    pre_pro = Preprocess()

    X, y = read_sup_dataset(tweet6k_path)
    X, y = pre_pro.transform(X, y)

    data_100k = read_unsup_dataset(tweet100k_path)
    data_100k = pre_pro.transform(data_100k)

    data_6k = train_test_split(X, y, test_size=0.2, stratify=y)

    params = {
        'c_list': [0.1, 2, 5, 10],
        'tol': [1e-4, 1e-5],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'linear_svm',
        'load': True
    }

    model6k = train_model(data_6k, **params)
    # plot_roc_curve(model6k, data)




if __name__ == '__main__':
    main()
