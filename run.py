import os


def read_sup_dataset(path):
    labels = ['positive', 'negative', 'notr']
    data_dict = {}
    for label in labels:
        data_path = os.path.join(path, label + '.txt')
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data.append(line)
        data_dict[label] = data
    return data_dict


def main():
    tweet6k_path = 'dataset/twitter_6K/'
    data = read_sup_dataset(tweet6k_path)
    print()


if __name__ == '__main__':
    main()