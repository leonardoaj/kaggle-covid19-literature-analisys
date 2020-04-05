from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from functions import get_literature_as_list, get_filename_list
import os
import pickle
import matplotlib.pyplot as plt
import shutil

if "kmeans.p" not in os.listdir("."):
    vectorizer = TfidfVectorizer(use_idf=True)

    literature = get_literature_as_list()

    if "vectors.p" not in os.listdir("."):
        print("Vectorizing literature")
        vectors = vectorizer.fit_transform(literature)
        pickle.dump(vectors, open("vectors.p", "wb"))
    else:
        vectors = pickle.load(open("vectors.p", "rb"))


    print("Fitting kmeans")

    kmeans = MiniBatchKMeans(n_clusters=21,
                             random_state=0,
                             batch_size=2000,
                             max_iter=3000).fit(vectors)
    # sse = {}
    # for k in range(5, 25):
    #     kmeans = MiniBatchKMeans(n_clusters=k,
    #                             random_state=0,
    #                             batch_size=2000,
    #                             max_iter=2000).fit(vectors)
    #     # sse[k] = kmeans.inertia_
    #     # sil_coeff = silhouette_score(vectors, kmeans.labels_, metric='euclidean')
    #     # print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))
    #
    # plt.figure()
    # plt.plot(list(sse.keys()), list(sse.values()))
    # plt.xlabel("Number of clusters")
    # plt.ylabel("SSE")
    # plt.show()
    # plt.savefig("elbow.png")

    # kmeans = KMeans(n_clusters=2,
    #                 random_state=0,
    #                 max_iter=2000).fit(vectors)

    pickle.dump(kmeans, open("kmeans.p", "wb"))

# feature_names = vectorizer.get_feature_names()
#
# df = pd.DataFrame(vectors.todense().tolist(), columns=feature_names)
#
# pickle.dump(df, open("dataframe.p", "wb"))

print("Finished.")
