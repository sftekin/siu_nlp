import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, path=''):
        self.model_path = os.path.join(path, 'word2vec.pkl')
        self.vector_path = os.path.join(path, 'word2vec.vec')

        if os.path.isfile(self.model_path):
            self.model = pickle.load(self.model_path)
        else:
            self.model = KeyedVectors.load_word2vec_format(self.vector_path, binary=False, unicode_errors='replace')
            pickle.dump(self.model, self.model_path)

        self.vector_size = self.model.vector_size

        out_of_vocab_vector = np.random.rand(1, self.vector_size)[0]
        out_of_vocab_vector = out_of_vocab_vector - np.linalg.norm(out_of_vocab_vector)

        self.out_of_vocab_vector = out_of_vocab_vector

    def fit(self, *_):
        return self

    def transform(self, X):
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

    word2vec = MeanEmbedding()

    vectors = word2vec.fit_transform(tokens)
    print(vectors)