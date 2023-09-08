from gensim.models import Phrases
from gensim.test.utils import common_texts
from gensim.utils import tokenize
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import gensim.downloader as api
from prep_names import organize_subj_df

import numpy as np
import pandas as pd
import string
import gensim
from gensim.models import phrases, word2vec
from pickle_wrap import pickle_wrap
from tqdm import tqdm
from pprint import pprint
import pickle
from wordvec_bigrams import get_bigrams


def get_bigram_vectors(parts_do=15, n_parts=100):
    corpus = []
    for i in range(parts_do):
        print(f'Loading part {i} of {parts_do}')
        fp_part = f'cache/wiki_corpus_parts/schemerep_bigram_corpus_wiki_part{i}_{n_parts}.pkl'
        with open(fp_part, 'rb') as f:
            corpus += pickle.load(f)
    print('Starting model...')
    import logging
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    model = Word2Vec(corpus, min_count=1, epochs=1)
    fp_bigrams = r'cache/schemerep_bigrams.pkl'
    bigrams = pickle_wrap(fp_bigrams, lambda: get_bigrams(),
                          easy_override=True)
    for bigram in bigrams:
        bigram = '_'.join(bigram)
        try:
            print(f'{bigram} | {model.wv.most_similar(bigram)}')
        except KeyError:
            print(f'Not found: {bigram}')
        # if bigram in model.wv:
        #     print(f'Found bigram: {bigram}')
        # else:
        #     print(f'Not found: {bigram}')

def fit_word2vec(parts_do=20, n_parts=100, min_count=5, epochs=5):
    corpus = []
    for i in range(parts_do):
        print(f'Loading part {i} of {parts_do}')
        fp_part = f'cache/wiki_corpus_parts/schemerep_bigram_corpus_wiki_part{i}_{n_parts}.pkl'
        with open(fp_part, 'rb') as f:
            corpus += pickle.load(f)
    print('Starting model...')
    import logging
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    model = Word2Vec(corpus, min_count=min_count, epochs=epochs)
    return model




if __name__ == '__main__':
    # get_bigram_vectors()
    fp_model = r'cache/schemerep_word2vec_model_first.pkl'
    model = pickle_wrap(fp_model, lambda: fit_word2vec(), easy_override=True)


