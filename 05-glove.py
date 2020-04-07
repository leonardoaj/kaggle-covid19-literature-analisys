import itertools
# import multiprocessing
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
from functions import get_literature_as_list
import pickle
import os

# sentences = list(itertools.islice(Text8Corpus('text8'), None))
print("Loading literature")
if not "sentence_vectors.p" in os.listdir("."):
    sentences = get_literature_as_list(stemmize=True,
                                       sentences_as_list=True,
                                       enable_multiprocessing=False,
                                       split_sentences=True)
else:
    sentences = pickle.load(open("sentence_vectors.p", "rb"))

# fitting the corpus with sentences and creating Glove object
print("Fitting corpus")
corpus = Corpus()
corpus.fit(sentences, window=10)
pickle.dump(corpus, open("corpus.p", "wb"))

# fitting to the corpus and adding standard dictionary to the object
print("Calculating glove")
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=8, verbose=True)

glove.add_dictionary(corpus.dictionary)
pickle.dump(glove, open("glove.p", "wb"))

