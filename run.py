import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import Preprocess
from train import train_model
from config import model_config
from embedding import Embedding
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

    max_seqlen = np.max([len(twt) for twt in X])

    # Create the word2vec object
    word2vec = Embedding(seq_len=max_seqlen)
    X_vectors = word2vec.transform(X)

    # concat external features to end of embeddings
    X_vectors = np.concatenate((X_vectors, features), axis=1)

    data_20k = train_test_split(X_vectors, y, test_size=0.2, stratify=y, random_state=42)





if __name__ == '__main__':
    main()
