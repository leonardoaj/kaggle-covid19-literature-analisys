from gensim.summarization.bm25 import get_bm25_weights
from functions import get_filename_list, read_json
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pickle
import multiprocessing


def stemm_paragraph(paragraph):
    return [ps.stem(word) for word in word_tokenize(paragraph.lower())]


def process_corpus(files, output, rank):
    corpus = []

    for k, json_file in enumerate(files):
        paragraphs = [stemm_paragraph(x) for x in json_file]
        corpus.extend(paragraphs)

    output[rank] = corpus

ps = PorterStemmer()

files = get_filename_list()
total_files = len(files)
num_cores = multiprocessing.cpu_count()
chunk_size = total_files // num_cores

processes = []
manager = multiprocessing.Manager()
output = manager.dict()

for rank in range(num_cores):

    if rank + 1 == num_cores:
        file_chunk = files[rank*chunk_size:]
    else:
        file_chunk = files[rank*chunk_size: (rank+1)*chunk_size]

    print(f"Reading chunk {rank}...")
    json_contents_list = [read_json(x, as_list=True) for x in file_chunk]

    p = multiprocessing.Process(target=process_corpus, args=(json_contents_list,
                                                          output,
                                                          rank))
    p.start()
    processes.append(p)

for k, p in enumerate(processes):
    p.join()
    print(f"{k} has finished")

main_corpus = []

for k, v in output:
    main_corpus.extend(v)

bm25_weights = get_bm25_weights(main_corpus, n_jobs=num_cores)

pickle.dump(bm25_weights, open("bm25_weights.p", "wb"))


