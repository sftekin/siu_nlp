import numpy as np
from run_helper import read_sup_dataset, preprocess_set, split_data


def main():
    tweet20k_path = 'dataset/twitter_20K'

    # Supervised data
    X, y = read_sup_dataset(tweet20k_path)

    # preprocess data
    X, y, int2word, word2int = preprocess_set(X, y)

    # split data
    data_dict = split_data(X, y)


    # Create the word2vec object
    # word2vec = Embedding(seq_len)
    # X_vectors = word2vec.transform(X)
    #
    # # concat external features to end of embeddings
    # X_vectors = np.concatenate((X_vectors, features), axis=1)
    #
    # data_20k = train_test_split(X_vectors, features, y, test_size=0.2, stratify=y, random_state=42)
    # print()


if __name__ == '__main__':
    main()
