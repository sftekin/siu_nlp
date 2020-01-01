import re
import nltk
import emoji
import string

from nltk.tokenize import TweetTokenizer
# nltk.download('stopwords')
from nltk.corpus import stopwords

tur_stopwords = set(stopwords.words('turkish'))


class Preprocess:
    def __init__(self):
        self.tokenizer = TweetTokenizer()

        # remove url, long-mention, mention, hash-tag, emoji
        self.re_list = [r'http\S+', r'\(@ ?[^\s].+\)', r'@ ?[^\s]+', r'# ?[^\s]+']

        # before tokenization
        self.rmv_emoji = lambda x: emoji.get_emoji_regexp().sub(r'', x)

        # after tokenization
        self.rmv_stop = lambda x: [w for w in x if w not in tur_stopwords]
        self.rmv_pun = lambda x: [w for w in x if w not in string.punctuation]
        self.rmv_short_long = lambda x: [w for w in x if 15 >= len(w) > 3]

        self.rmv_list = [self.rmv_stop, self.rmv_pun, self.rmv_short_long]

    def transform(self, X, y=None):
        """
        :param X: [str, str, ..., str]
        :return:[[token , ..., token], ..., [token , ..., token]]
        """
        clean_data = []
        if y:
            labels = []
        for idx, sen in enumerate(X):
            tokens = self._preprocess(sen)
            if len(tokens) > 0:
                clean_data.append(tokens)
                if y:
                    labels.append(y[idx])
        if y:
            return_val = clean_data, labels
        else:
            return_val = clean_data
        return return_val

    def _preprocess(self, sentence):
        """
        :param sentence: string
        :return: [token, ..., token]
        """
        for re_op in self.re_list:
            sentence = re.sub(re_op, '', sentence)
        sentence = self.rmv_emoji(sentence)

        tokens = self.tokenizer.tokenize(sentence)

        for rmv_op in self.rmv_list:
            tokens = rmv_op(tokens)

        return tokens


