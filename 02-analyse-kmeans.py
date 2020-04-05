from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import pickle
import matplotlib.pyplot as plt
import shutil
import sys

print("Fitting kmeans")

sse = {}
sil_coeffs = {}

vectors = pickle.load(open("vectors.p", "rb"))

for k in range(5, 25):
    kmeans = MiniBatchKMeans(n_clusters=k,
                            random_state=0,
                            batch_size=2000,
                            max_iter=2000).fit(vectors)
    sse[k] = kmeans.inertia_
    if len(sys.argv) > 1 and sys.argv[1] == "sil":
        sil_coeffs[k] = silhouette_score(vectors, kmeans.labels_, metric='euclidean')


plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.savefig("elbow.png")

max_value = max(sse.values())
best_k = -1
for k, v in sse.items():
    if v == max_value:
        best_k = k
        break

kmeans = MiniBatchKMeans(n_clusters=best_k,
                        random_state=0,
                        batch_size=2000,
                        max_iter=2000).fit(vectors)

pickle.dump(kmeans, open("kmeans.p", "wb"))
