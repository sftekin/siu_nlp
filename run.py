"""
"""
from sklearn.model_selection import train_test_split
from preprocessing import Preprocess
from train import train_model
from run_helper import plot_roc_curve, read_unsup_dataset, \
    read_sup_dataset, self_label


def main():
    tweet100k_path = 'dataset/twitter_100K'
    tweet6k_path = 'dataset/twitter_6K'

    pre_pro = Preprocess()

    X_sm, y_sm = read_sup_dataset(tweet6k_path, pre_pro)
    data_6k = train_test_split(X_sm, y_sm, test_size=0.2, stratify=y_sm)
    X_train, X_test, y_train, y_test = data_6k

    data_100k = read_unsup_dataset(tweet100k_path, pre_pro, sample_size=100000)


    params = {
        'n_estimator': [10, 100, 300, 500],
        'max_depth': [50, 100, 300],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'random_forest',
        'load': False
    }

    model6k = train_model(data_6k, **params)
    thresholds = plot_roc_curve(model6k, data_6k, fig_name='roc_6k')

    X_big, y_big = self_label(model6k, data_100k, **thresholds)

    # Merge data
    X, y = X_big + X_train, y_big + y_train
    data_106k = train_test_split(X, y, test_size=0.2, stratify=y)

    params = {
        'n_estimator': [10, 100, 300, 500],
        'max_depth': [3, 6, 9],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'random_forest_big',
        'load': False
    }

    model106k = train_model(data_106k, **params)
    thresholds = plot_roc_curve(model106k, data_6k, fig_name='roc_100k')

    # test on first dataset
    score_original = model6k.score(X_test, y_test)
    score_self_learned = model106k.score(X_test, y_test)
    print('Original model score on test_data:{}\n'
          'Self learned model score on test_data:{}'.format(score_original, score_self_learned))


if __name__ == '__main__':
    main()
