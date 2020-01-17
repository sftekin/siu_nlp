from run_helper import read_sup_dataset, preprocess_set, split_data
from batch_generator import BatchGenerator
from config import batch_params


def main():
    tweet20k_path = 'dataset/twitter_20K'

    # Supervised data
    X, y = read_sup_dataset(tweet20k_path)

    # preprocess data
    X, y, features, int2word, word2int = preprocess_set(X, y)

    # split data
    data_dict = split_data(X, y, features)

    batch_gen = BatchGenerator(data_dict, **batch_params)


if __name__ == '__main__':
    main()
