import io
import pickle


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])

    embed_path = open('embed.pkl', 'rb')
    pickle.dump(embed_path, data)
    return data


load_vectors(fname='cc.tr.300.bin')

