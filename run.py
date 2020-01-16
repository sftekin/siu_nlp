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
    tweet20k_path = 'dataset/twitter_20K'

    pre_pro = Preprocess()

    # Load Data sets
    # Supervised data
    (X, y), features = read_sup_dataset(tweet20k_path, pre_pro)

    # Create the word2vec object
    word2vec = MeanEmbedding()
    X_vectors = word2vec.transform(X)

    data_6k = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


if __name__ == '__main__':
    main()
