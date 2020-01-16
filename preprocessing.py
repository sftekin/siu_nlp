import re
import emoji
import string
import numpy as np

from nltk.tokenize import TweetTokenizer
# nltk.download('stopwords')
from nltk.corpus import stopwords

tur_stopwords = set(stopwords.words('turkish'))


class Preprocess:
    def __init__(self):
        self.tokenizer = TweetTokenizer()

        # remove url, long-mention, mention, hash-tag
        self.re_list = [r'http\S+', r'\(@ ?[^\s].+\)', r'@ ?[^\s]+', r'# ?[^\s]+']

        # before tokenizing, remove non textual emoji
        self.rmv_emoji = lambda x: emoji.get_emoji_regexp().sub(r'', x)

        # textual emoji
        self.pattern_happy_emoji = re.compile(r'([:;=] ?-?(\)|D+|d+))|(\(:)')
        self.pattern_unhappy_emoji = re.compile(r'(: ?-?(\(|/|([Ss]))+)')

        # after tokenizing
        self.rmv_stop = lambda x: [w for w in x if w not in tur_stopwords]
        self.rmv_pun = lambda x: [w for w in x if w not in string.punctuation]
        self.rmv_short_long = lambda x: [w for w in x if 20 >= len(w) >= 3]

        self.rmv_list = [self.rmv_stop, self.rmv_pun, self.rmv_short_long]

    def transform(self, X, y=None):
        """
        :param X: [str, str, ..., str]
        :return:[[token , ..., token], ..., [token , ..., token]]
        """
        clean_data = []
        extracted_features = []
        if y:
            labels = []
        for idx, sen in enumerate(X):
            tokens, feature = self._preprocess(sen)
            if len(tokens) > 0:
                clean_data.append(tokens)
                extracted_features.append(feature)
                if y:
                    labels.append(y[idx])
        if y:
            return_val = clean_data, labels
        else:
            return_val = clean_data
        return return_val, extracted_features

    def _preprocess(self, sentence):
        """
        :param sentence: string
        :return: [token, ..., token]
        """
        # remove url, long-mention, mention, hash-tag, non-textual emoji
        for re_op in self.re_list:
            sentence = re.sub(re_op, '', sentence)
        sentence = self.rmv_emoji(sentence)

        # extract textual-emoji feature
        feature = self._extract_feature(sentence)

        # tokenize the sentence
        tokens = self.tokenizer.tokenize(sentence)
        for rmv_op in self.rmv_list:
            tokens = rmv_op(tokens)
        return tokens, feature

    def _extract_feature(self, sentence):
        emoji_flag = np.array([0, 0])

        # search for happy emoji
        if self.pattern_happy_emoji.search(sentence):
            emoji_flag[0] = 1

        # search for unhappy emoji
        if self.pattern_unhappy_emoji.search(sentence):
            emoji_flag[1] = 1
        return emoji_flag


