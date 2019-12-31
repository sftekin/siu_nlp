import re
import nltk
import string

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import TweetTokenizer

try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

tur_stopwords = set(stopwords.words('turkish'))


class Preprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = TweetTokenizer()

        # remove url, mention, hash-tag
        self.re_list = [r'http\S+', r'@ ?[^\s]+', r'# ?[^\s]+']

        # after tokenization
        self.rmv_stop = lambda x: [w for w in x if w not in tur_stopwords]
        self.rmv_pun = lambda x: [w for w in x if w not in string.punctuation]
        self.rmv_short = lambda x: [w for w in x if len(w) < 3]

        self.rmv_list = [self.rmv_stop, self.rmv_pun, self.rmv_short]

    def fit(self, *_):
        return self

    def transform(self, X):
        """
        :param X: [str, str, ..., str]
        :return:[[token , ..., token], ..., [token , ..., token]]
        """
        clean_data = []
        for sen in X:
            tokens = self._preprocess(sen)
            if len(tokens) > 0:
                clean_data.append(tokens)

    def _preprocess(self, sentence):
        """
        :param sentence: string
        :return: [token, ..., token]
        """
        for re_op in self.re_list:
            sentence = re.sub(re_op, '', sentence)

        tokens = self.tokenizer.tokenize(sentence)

        for rmv_op in self.rmv_list:
            tokens = rmv_op(tokens)

        return tokens


