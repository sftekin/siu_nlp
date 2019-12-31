import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin


def load_vector():
    model_path = 'embedding/word2vec.pkl'
    vector_path = 'embedding/word2vec.vec'

    if os.path.isfile(model_path):
        model_file = open(model_path, 'rb')
        model = pickle.load(model_file)
    else:
        model = KeyedVectors.load_word2vec_format(vector_path, binary=False, unicode_errors='replace')
        model_file = open(model_path, 'wb')
        pickle.dump(model, model_file)

    return model


class MeanEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None
        self.vector_size = None
        self.out_of_vocab_vector = None

    def set_params(self, **params):
        self.model = params['model']
        self.vector_size = self.model.vector_size

        out_of_vocab_vector = np.random.rand(1, self.vector_size)[0]
        out_of_vocab_vector = out_of_vocab_vector - np.linalg.norm(out_of_vocab_vector)

        self.out_of_vocab_vector = out_of_vocab_vector
        return self

    def fit(self, *_):
        return self

    def transform(self, X):
        data = []
        for sen in X:
            data.append(self.vec2sen(sen))
        return np.stack(data, axis=0)

    def vec2sen(self, X):
        res = np.zeros(self.vector_size)
        for x in X:
            if x in self.model.wv:
                r = self.model.wv[x]
            elif x.lower() in self.model.wv:
                r = self.model.wv[x.lower()]
            else:
                r = self.out_of_vocab_vector
            res += r
        return res / len(X)


if __name__ == '__main__':

    tokens = 'bir kac kisi geldi sadece'.split()
    model = load_vector()
    word2vec = MeanEmbedding()

    vectors = word2vec.fit_transform(tokens)
    print(vectors)