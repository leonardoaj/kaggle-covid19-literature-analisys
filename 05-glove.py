import itertools
import multiprocessing
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
from functions import get_literature_as_list
import pickle

# sentences = list(itertools.islice(Text8Corpus('text8'), None))
sentences = get_literature_as_list(stemmize=True, sentences_as_list=True)

# fitting the corpus with sentences and creating Glove object
corpus = Corpus()
corpus.fit(sentences, window=10)
pickle.dump(corpus, open("corpus.p", "wb"))

# fitting to the corpus and adding standard dictionary to the object
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=multiprocessing.cpu_count(), verbose=True)

glove.add_dictionary(corpus.dictionary)
pickle.dump(glove, open("glove.p", "wb"))

