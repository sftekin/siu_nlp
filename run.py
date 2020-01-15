from sklearn.model_selection import train_test_split
from preprocessing import Preprocess
from train import train_model
from config import model_config
from embedding import MeanEmbedding
from run_helper import plot_roc_curve, read_unsup_dataset, \
    read_sup_dataset, self_label, compare_models


def main():
    tweet100k_path = 'dataset/twitter_100K'
    tweet6k_path = 'dataset/twitter_6K'

    pre_pro = Preprocess()

    # Load Data sets
    # Supervised data
    X, y = read_sup_dataset(tweet6k_path, pre_pro)
    # Unsupervised data
    data_100k = read_unsup_dataset(tweet100k_path, pre_pro,
                                   sample_size=100000, load=True)

    data_6k = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_test, y_train, y_test = data_6k

    # Create the word2vec object
    word2vec = MeanEmbedding()

    # Store the initial performance of models
    model_1 = train_model(data_6k, word2vec, **model_config['LinSVM'])
    model_2 = train_model(data_6k, word2vec, **model_config['RandomForest'])

    for model in [model_1, model_2]:
        score_original = model.score(word2vec.transform(X_test), y_test)
        # plot_roc_curve(model, data_6k, fig_name='roc_6k')
        print(score_original)

    # label data
    thresholds = [1, 0.85]
    X_big_1, y_big_1 = self_label(model_1.decision_function,
                                  word2vec,
                                  data_100k,
                                  thresholds[0])
    X_big_2, y_big_2 = self_label(model_2.predict_proba,
                                  word2vec,
                                  data_100k,
                                  thresholds[1])

    # Merge data
    X, y = X_big_1 + X_big_2 + X_train, y_big_1 + y_big_2 + y_train

    for model in [model_1, model_2]:
        print('Training the model with merged data')
        word_embeds = word2vec.transform(X)
        model = model.fit(word_embeds, y)
        # plot_roc_curve(model, data_6k, fig_name='roc_100k')

        # test on first dataset
        score_self_learned = model.score(word2vec.transform(X_test),
                                         y_test)
        print('Original model average '
              'precision score on test_data:{}\n'
              'Self learned model average '
              'precision score on test_data:{}'.format(score_original, score_self_learned))

    # compare_models([model, model106k], X_test, y_test, ['6k', '10k'])


if __name__ == '__main__':
    main()
