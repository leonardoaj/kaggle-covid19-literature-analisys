from wordcloud import WordCloud
from nltk.corpus import stopwords
import os
import matplotlib.pyplot as plt
from functions import get_literature_as_list, get_vectorizer
import multiprocessing


def generate_word_cloud(group):
    files = [f"groups/{group}/{x}" for x in os.listdir(f"groups/{group}")]

    preprocessed_papers = get_literature_as_list(files=files, stemmize=False, enable_multiprocessing=False)

    vectors, vectorizer = get_vectorizer(preprocessed_papers)

    freqs = {word: vectors.getcol(idx).sum() for word, idx in vectorizer.vocabulary_.items()}

    cloud = wc.fit_words(frequencies=freqs)

    plt.figure(figsize=(16, 16), facecolor=None)
    plt.imshow(cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"wordcloud-{group}.png")


stop_words = stopwords.words('english')

wc = WordCloud(
    background_color="white",
    max_words=2000,
    width=1024,
    height=720,
    stopwords=stop_words
)

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
pool.map(generate_word_cloud, [str(x).zfill(2) for x in os.listdir("groups")])
