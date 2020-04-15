import pandas as pd
import os
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, words
import re
import pickle
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

ps = PorterStemmer()

try:
    stop_words = stopwords.words('english')
    valid_words = set([ps.stem(x) for x in words.words()])
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('words')
    stop_words = stopwords.words('english')


def get_filename_list():

    if not "filename_list.p" in os.listdir("."):

        directories = [x[0] for x in os.walk("data")]

        files = []

        for dir in directories[1:]:
            files.extend([f"{dir}/{x}" for x in os.listdir(dir) if os.path.isfile(f"{dir}/{x}")])

        pickle.dump(files, open("filename_list.p", "wb"))

    else:
        files = pickle.load(open("filename_list.p", "rb"))

    return files


def pre_process(files, output, rank, stemmize, sentences_as_list):

    text_list = []

    for k, full_body_text in enumerate(files):

        full_body_text = re.sub('[\_\.\,\-\(\)\[\]\/\d\+\"\=]', "", full_body_text)

        if stemmize:
            full_body_text = " ".join([ps.stem(x) for x in full_body_text.split() if x not in stop_words])
        else:
            full_body_text = " ".join([x for x in full_body_text.split() if x not in stop_words])

        if sentences_as_list:
            full_body_text = [x for x in full_body_text.split() if x in valid_words]
        else:
            full_body_text = " ".join([x for x in full_body_text.split() if x in valid_words])

        text_list.append(full_body_text)

    output[rank] = text_list

def read_json(file, as_list=False):
    json_file = json.loads(open(file).read())
    if as_list:
        return [x['text'] for x in json_file['body_text']]
    else:
        return "\n".join([x['text'] for x in json_file['body_text']])



def get_vectorizer(literature):

    vectorizer = TfidfVectorizer(use_idf=True)

    print("Vectorizing literature")
    vectors = vectorizer.fit_transform(literature)

    return vectors, vectorizer


def get_literature_as_list(files=None,
                           stemmize=True,
                           sentences_as_list=False,
                           load_pickle=False,
                           enable_multiprocessing=True,
                           split_sentences=False,
                           save_pickle=True) -> list:

    if load_pickle and "text_list.p" in os.listdir("."):
        text_list = pickle.load(open("text_list.p", "rb"))
    else:

        text_list = []
        files = get_filename_list() if not files else files
        total_files = len(files)
        num_cores = multiprocessing.cpu_count()
        chunk_size = total_files // num_cores

        if enable_multiprocessing:
            processes = []
            manager = multiprocessing.Manager()
            output = manager.dict()

            for rank in range(num_cores):

                if rank + 1 == num_cores:
                    file_chunk = files[rank*chunk_size:]
                else:
                    file_chunk = files[rank*chunk_size: (rank+1)*chunk_size]

                print(f"Reading chunk {rank}...")
                json_contents_list = [read_json(x) for x in file_chunk]

                p = multiprocessing.Process(target=pre_process, args=(json_contents_list,
                                                                      output,
                                                                      rank,
                                                                      stemmize,
                                                                      sentences_as_list))
                p.start()
                processes.append(p)

            for k, p in enumerate(processes):
                p.join()
                print(f"{k} has finished")

        else:
            rank = 0
            output = {}
            json_contents_list = [read_json(x) for x in files]
            pre_process(json_contents_list, output, rank, stemmize, split_sentences)

        for k, v in output.items():
            text_list.extend(v)

        if split_sentences and save_pickle:
            pickle.dump(text_list, open("sentence_vectors.p", "wb"))

        if save_pickle:
            pickle.dump(text_list, open("text_list.p", "wb"))

    return text_list


