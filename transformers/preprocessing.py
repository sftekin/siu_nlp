import re
import emoji
import string
import nltk

from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords

tur_stopwords = set(stopwords.words('turkish'))


class Preprocess:
    def __init__(self, sequence_len):
        self.sequence_len = sequence_len
        self.tokenizer = TweetTokenizer()

        # remove url, long-mention, mention, hash-tag, numbers
        self.re_list = [r'http\S+', r'\(@ ?[^\s].+\)', r'@ ?[^\s]+', r'# ?[^\s]+', r'[0-9]+']

        # before tokenizing, remove non textual emoji
        self.rmv_emoji = lambda x: emoji.get_emoji_regexp().sub(r'', x)

        # replace repetitive chars with only one char
        self.re_repetitive = r'(.)\1{2,}'
        self.rep_pattern = re.compile(r'(.)\1{2,}')

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
        if y:
            labels = []
        for idx, sen in enumerate(X):
            tokens = self._preprocess(sen)
            if 0 < len(tokens) < self.sequence_len:
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
        # remove url, long-mention, mention, hash-tag, non-textual emoji
        for re_op in self.re_list:
            sentence = re.sub(re_op, '', sentence)
        sentence = self.rmv_emoji(sentence)

        # replace repetitive chars
        all_match = self.rep_pattern.findall(sentence)
        for match in all_match:
            sentence = re.sub(self.re_repetitive, match, sentence, count=1)

        # tokenize the sentence
        tokens = self.tokenizer.tokenize(sentence)
        for rmv_op in self.rmv_list:
            tokens = rmv_op(tokens)

        # lower every word
        tokens = [token.lower() for token in tokens]

        return tokens
