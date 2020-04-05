import pickle
from functions import get_filename_list
import os
import shutil

kmeans = pickle.load(open("kmeans.p", "rb"))
vectorized_papers = pickle.load(open("vectors.p", "rb"))

filenames = get_filename_list()

if not "groups" in os.listdir("."):
    for i in range(0, 21):
        os.makedirs(f"groups/{str(i).zfill(2)}")

for file, paper_vector in zip(filenames, vectorized_papers):
    group = kmeans.predict(paper_vector)
    shutil.copy(file, f"groups/{str(group[0]).zfill(2)}")

print("Done.")