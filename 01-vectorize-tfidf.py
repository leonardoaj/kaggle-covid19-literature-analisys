from functions import get_literature_as_list
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

vectorizer = TfidfVectorizer(use_idf=True)

literature = get_literature_as_list()

print("Vectorizing literature")
vectors = vectorizer.fit_transform(literature)
pickle.dump(vectors, open("vectors.p", "wb"))
pickle.dump(vectorizer, open("vectorizer.p", "wb"))