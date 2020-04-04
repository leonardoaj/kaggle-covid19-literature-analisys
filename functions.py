import pandas as pd
import os
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle

ps = PorterStemmer()
stop_words = stopwords.words('english')

def get_literature_df()->pd.DataFrame:

    df = pd.DataFrame()

    files = get_filename_list()

    for file in files:
        df_aux = pd.DataFrame()

    return df


def get_filename_list():

    directories = ["data/biorxiv_medrxiv",
                   "data/comm_use_subset",
                   "data/noncomm_use_subset",
                   "data/pmc_custom_license"]

    files = []

    for dir in directories:
        files.extend([f"{dir}/{x}" for x in os.listdir(dir)])

    return files


def get_literature_as_list(preprocess=False) -> list:

    try:
        text_list = pickle.load(open("text_list.p", "rb"))
    except:
        text_list = []
        files = get_filename_list()
        total_files = len(files)

        for k, file in enumerate(files):

            print(k/total_files)

            literature_obj = json.loads(open(file).read())
            full_body_text = "\n".join([x['text'] for x in literature_obj['body_text']])

            if preprocess:
                full_body_text = re.sub('[\.\,\-\(\)\[\]\/\d\+\"\=]', "", full_body_text)
                full_body_text = " ".join([ps.stem(x) for x in full_body_text.split() if x.lower() not in stop_words and len(x) <= 20])

            text_list.append(full_body_text)

        pickle.dump(text_list, open("text_list.p", "wb"))

    return text_list


