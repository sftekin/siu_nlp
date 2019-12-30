import numpy as np
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=False, unicode_errors='replace')
        self.vector_size = self.model.vector_size

        out_of_vocab_vector = np.random.rand(1, self.vector_size)[0]
        out_of_vocab_vector = out_of_vocab_vector - np.linalg.norm(out_of_vocab_vector)

        self.out_of_vocab_vector = out_of_vocab_vector

    def fit_transform(self, X, y=None, **kwargs):
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

    word2vec = MeanEmbedding('word2vec.vec')
    tokens = 'bir kac kisi geldi sadece'.split()

    vectors = word2vec.fit_transform(tokens)
    print(vectors)

