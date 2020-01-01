"""
"""
from sklearn.model_selection import train_test_split
from preprocessing import Preprocess
from train import train_model
from run_helper import plot_roc_curve, read_unsup_dataset, \
    read_sup_dataset, self_label, compare_models


def main():
    tweet100k_path = 'dataset/twitter_100K'
    tweet6k_path = 'dataset/twitter_6K'

    pre_pro = Preprocess()

    X, y = read_sup_dataset(tweet6k_path, pre_pro)
    data_6k = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_test, y_train, y_test = data_6k

    data_100k = read_unsup_dataset(tweet100k_path, pre_pro, sample_size=10000, load=False)

    params = {
        'n_estimator': [1000],
        'max_depth': [5000],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'random_forest',
        'load': True
    }

    model = train_model(data_6k, **params)
    score_original = model.score(X_test, y_test)
    print(score_original)
    thresholds = plot_roc_curve(model, data_6k, fig_name='roc_6k')
    print(model.score(X_test, y_test))

    # label data
    X_big, y_big = self_label(model, data_100k, **thresholds)

    # Merge data
    X, y = X_big + X_train, y_big + y_train

    print('Training the model with merged data')
    model = model.fit(X, y)
    thresholds = plot_roc_curve(model, data_6k, fig_name='roc_100k')

    # test on first dataset
    score_self_learned = model.score(X_test, y_test)
    print('Original model average precision score on test_data:{}\n'
          'Self learned model average precision score on test_data:{}'.format(score_original, score_self_learned))

    # compare_models([model, model106k], X_test, y_test, ['6k', '10k'])


if __name__ == '__main__':
    main()
