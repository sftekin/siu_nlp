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

    data_100k = read_unsup_dataset(tweet100k_path, pre_pro)


    params = {
        'c_list': [0.1, 2, 5, 10],
        'tol': [1e-4, 1e-5],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'linear_svm',
        'load': True
    }

    model6k = train_model(data_6k, **params)
    thresholds = plot_roc_curve(model6k, data_6k, fig_name='roc_6k')

    X_big, y_big = self_label(model6k, data_100k, **thresholds)

    # Merge data
    X, y = X_big + X_sm, y_big + y_sm
    data_106k = train_test_split(X, y, test_size=0.2, stratify=y)

    params = {
        'c_list': [0.1, 2, 5, 10],
        'tol': [1e-4, 1e-5],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'linear_svm_big',
        'load': True
    }

    model106k = train_model(data_106k, **params)
    thresholds = plot_roc_curve(model106k, data_6k, fig_name='roc_100k')

    # test on first dataset
    _, X_test, _, y_test = data_6k
    score_original = model6k.score(X_test, y_test)
    score_self_learned = model106k.score(X_test, y_test)
    print('Original model score on test_data:{}\n'
          'Self learned model score on test_data:{}'.format(score_original, score_self_learned))


if __name__ == '__main__':
    main()
