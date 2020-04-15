import pickle
from nltk import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, words
import json


def stemm_question(question):
    return list(set([ps.stem(word) for word in word_tokenize(" ".join(regex.tokenize(question))) if ps.stem(word) in valid_words]))


ps = PorterStemmer()
regex = RegexpTokenizer(r"\w+")
valid_words = set([ps.stem(x) for x in words.words()])

index = pickle.load(open("inverted-index/inverted_index_aux.p", "rb"))

questions = [
    "Are there geographic variations in the rate of COVID-19 spread?",
    "Are there geographic variations in the mortality rate of COVID-19?",
    "Is there any evidence to suggest geographic based virus mutations?"
]

vectors = pickle.load(open("vectors.p", "rb"))

question_tokens = stemm_question(" ".join(questions))

paper_score = {}

results = set(index[question_tokens[0]].keys())

for token in question_tokens[1:]:

    papers = index[token].keys()

    for paper in papers:

        if not paper in paper_score.keys():
            paper_score[paper] = 0

        paper_score[paper] += 1

    results.update(papers)

ordered_papers = {k: v for k, v in sorted(paper_score.items(), key=lambda item: item[1], reverse=True)}

chosen_paper = json.loads(open(list(ordered_papers.keys())[0]).read())
