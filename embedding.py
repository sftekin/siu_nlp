import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin


class Embedding(BaseEstimator, TransformerMixin):
    def __init__(self, seq_len):
        self.seq_len = seq_len

        self.model_path = 'embedding/word2vec.pkl'
        self.vector_path = 'embedding/word2vec.vec'

        if os.path.isfile(self.model_path):
            model_file = open(self.model_path, 'rb')
            self.model = pickle.load(model_file)
        else:
            self.model = KeyedVectors.load_word2vec_format(self.vector_path, binary=False, unicode_errors='replace')
            model_file = open(self.model_path, 'wb')
            pickle.dump(self.model, model_file)

        self.vector_size = self.model.vector_size

        out_of_vocab_vector = np.random.rand(1, self.vector_size)[0]
        out_of_vocab_vector = out_of_vocab_vector - np.linalg.norm(out_of_vocab_vector)

        self.out_of_vocab_vector = out_of_vocab_vector
        self.pad_vector = np.zeros((self.vector_size,))

    def fit(self, *_):
        return self

    def transform(self, X):
        data = []
        for sen in X:
            data.append(self._vec2sen(sen))
        return np.stack(data, axis=0)

    def _vec2sen(self, X):
        res = []
        for x in X:
            if x in self.model.wv:
                r = self.model.wv[x]
            elif x.lower() in self.model.wv:
                r = self.model.wv[x.lower()]
            else:
                r = self.out_of_vocab_vector
            res.append(r)
        if len(res) < self.seq_len:
            for _ in range(self.seq_len - len(res)):
                res.append(self.pad_vector)
        res = np.stack(res, axis=0)
        return res
