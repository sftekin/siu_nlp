import re
import nltk
import string

from nltk.tokenize import TweetTokenizer

try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

tur_stopwords = set(stopwords.words('turkish'))


class Preprocess:
    def __init__(self):
        self.tokenizer = TweetTokenizer()

        # remove url, long-mention, mention, hash-tag
        self.re_list = [r'http\S+', r'\(@ ?[^\s].+\)', r'@ ?[^\s]+', r'# ?[^\s]+']

        # after tokenization
        self.rmv_stop = lambda x: [w for w in x if w not in tur_stopwords]
        self.rmv_pun = lambda x: [w for w in x if w not in string.punctuation]
        self.rmv_short_long = lambda x: [w for w in x if 15 >= len(w) > 3]

        self.rmv_list = [self.rmv_stop, self.rmv_pun, self.rmv_short_long]

    def transform(self, X, y):
        """
        :param X: [str, str, ..., str]
        :return:[[token , ..., token], ..., [token , ..., token]]
        """
        clean_data = []
        labels = []
        for idx, sen in enumerate(X):
            tokens = self._preprocess(sen)
            if len(tokens) > 0:
                clean_data.append(tokens)
                labels.append(y[idx])
        return clean_data, labels

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


