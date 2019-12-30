import pickle
from gensim.models import FastText

ft_model = FastText.load('cc.tr.300.bin')

vocab = list(ft_model.wv.vocab)

word_to_vec_dict = {word: ft_model[word] for word in vocab}

with open('word2vec_dictionary.pickle', 'wb') as f:
    pickle.dump(word_to_vec_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print(word_to_vec_dict["word"])
