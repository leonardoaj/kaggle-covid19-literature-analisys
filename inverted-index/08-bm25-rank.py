from gensim.summarization.bm25 import get_bm25_weights
from functions import get_filename_list
import json
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pickle


def stemm_paragraph(paragraph):
    return [ps.stem(word) for word in word_tokenize(paragraph.lower())]


ps = PorterStemmer()

index = {}

corpus = []

for k, filename in enumerate(get_filename_list()):

    print(k, filename)
    json_file = json.loads(open(filename).read())
    paragraphs = [stemm_paragraph(x['text']) for x in json_file['body_text']]
    corpus.extend(paragraphs)

chunk = len(corpus) // 10

for i in range(9):
    pickle.dump(corpus[i*chunk: (i+1)*chunk], open(f"bm25_corpus_{i}.p", "wb"), pickle.HIGHEST_PROTOCOL)

bm25_weights = get_bm25_weights(corpus, n_jobs=10)

pickle.dump(bm25_weights, open("bm25_weights.p", "wb"))
