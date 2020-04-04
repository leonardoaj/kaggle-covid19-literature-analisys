from sklearn.feature_extraction.text import TfidfVectorizer
from functions import get_literature_as_list
import pandas as pd
import pickle

vectorizer = TfidfVectorizer()

literature = get_literature_as_list(preprocess=True)

print("Starting to vectorize...")
vectors = vectorizer.fit_transform(literature)
print("Done.")

feature_names = vectorizer.get_feature_names()

dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

pickle.dump(df, open("dataframe.p", "wb"))

print("done")