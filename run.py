from run_helper import read_sup_dataset, preprocess_set, split_data
from batch_generator import BatchGenerator
from train import train
from config import batch_params, train_params, model_config
from models.sentiment_model import SentimentModel


def main():
    tweet6k_path = 'dataset/twitter_6K'
    tweet20k_path = 'dataset/twitter_20K'

    # Supervised data
    X_1, y_1 = read_sup_dataset(tweet20k_path)
    X_2, y_2 = read_sup_dataset(tweet6k_path)

    # preprocess data
    X, y, int2word, word2int = preprocess_set(X_1+X_2, y_1+y_2)

    # split data
    data_dict = split_data(X, y)

    batch_gen = BatchGenerator(data_dict, **batch_params)

    model = SentimentModel(model_config, int2word)

    train(model, batch_gen, **train_params)


if __name__ == '__main__':
    main()
