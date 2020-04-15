from functions import get_filename_list
import json
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, words
import pickle


def stemm_paragraph(paragraph):
    return [ps.stem(word) for word in word_tokenize(paragraph)]


ps = PorterStemmer()
stop_words = stopwords.words('english')
valid_words = set([ps.stem(x) for x in words.words()])

index = {}

for k, filename in enumerate(get_filename_list()):

    print(k, filename)

    json_file = json.loads(open(filename).read())

    paragraphs = [x['text'] for x in json_file['body_text']]

    for i, paragraph in enumerate(paragraphs):

        stemmed_paragraph_tokens = stemm_paragraph(paragraph)

        for token in stemmed_paragraph_tokens:

            if not token in index.keys():
                index[token] = {}

            if not filename in index[token].keys():
                index[token][filename] = set()

            index[token][filename].add(i)

pickle.dump(index, open("inverted_index.p", "wb"))
