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

# bigrams = phrases.Phrases(sentences)

def add_bigrams_to_corpus(corpus, bigrams):
    for bigram in tqdm(bigrams, desc='Looping bigrams...'):
        corpus_new = []
        bigram_in = False
        for sentence in corpus:
            # pprint(sentence['section_texts'])
            onto_word = 0
            idxs_st = []
            idxs_end  = []
            for i, word in enumerate(sentence):
                if word == bigram[onto_word]:
                    onto_word += 1
                    if onto_word == len(bigram):
                        idxs_st.append(i - len(bigram) + 1)
                        idxs_end.append(i + 1)
                        onto_word = 0
                elif word == bigram[0]:
                    onto_word = 1
                else:
                    onto_word = 0
            if len(idxs_st):
                bigram_in = True
            for st, end in zip(idxs_st[::-1], idxs_end[::-1]):
                sentence = sentence[:st] + ['_'.join(bigram)] + sentence[end:]
            corpus_new.append(sentence)
        if bigram_in:
            print(f'{bigram} is in!')
        else:
            print(f'{bigram} not in :(')
        corpus = corpus_new
    return corpus

def get_bigrams():
    df = organize_subj_df('138')
    bigrams = []
    for obj, scene in zip(df['obj'], df['scene']):
        for s in [obj, scene]:
            if ' ' in s:
                bigrams.append(tuple(s.split(' ')))
    return bigrams

def tokenize_wiki_part(corpus_part):
    corpus_tokenized = []
    for article in tqdm(corpus_part, desc='Tokenizing wiki corpus'):
        for section_text in article['section_texts']:
            tokenized = tokenize(section_text, lower=True)
            corpus_tokenized.append(list(tokenized))
    return corpus_tokenized

def tokenize_wiki(corpus):
    print('Listing...')
    corpus = list(corpus)
    print('Listed')

    return corpus_tokenized

def get_tokenized_wiki():
    corpus = api.load('wiki-english-20171001')
    return tokenize_wiki(corpus)

def prep_bigram_corpus(corpus_name='text8'):
    fp_bigrams = r'cache/schemerep_bigrams.pkl'
    bigrams = pickle_wrap(fp_bigrams, lambda: get_bigrams())
    if corpus_name == 'wiki-english-20171001':
        corpus = get_tokenized_wiki()
    else:
        corpus = api.load(corpus_name)
    corpus = add_bigrams_to_corpus(corpus, bigrams)
    return corpus

def split_tokenize_bigram_wiki(n_parts=10):
    fp_bigrams = r'cache/schemerep_bigrams.pkl'
    bigrams = pickle_wrap(fp_bigrams, lambda: get_bigrams(),
                          easy_override=True)
    corpus = api.load('wiki-english-20171001')
    corpus = list(corpus)
    print('Listed')
    for i in range(n_parts):
        print(f'Tokenizing part {i} of {n_parts}')
        corpus_part = corpus[i::n_parts]
        fp_part = f'cache/schemerep_bigram_corpus_wiki_part{i}_{n_parts}.pkl'
        tokenized_part = pickle_wrap(fp_part,
                 lambda: add_bigrams_to_corpus(tokenize_wiki_part(corpus_part),
                                               bigrams))

def load_wiki():
    # https://github.com/RaRe-Technologies/gensim-data
    corpus = api.load('wiki-english-20171001')



def do(corpus_name='text8'):
    fp_bigrams = fr'cache/schemerep_bigram_corpus_{corpus_name}.pkl'
    corpus = pickle_wrap(fp_bigrams,
                         lambda: prep_bigram_corpus(corpus_name=corpus_name))
    model = Word2Vec(corpus, min_count=1)


if __name__ == '__main__':
    # get_bigram_vectors()
    split_tokenize_bigram_wiki(n_parts=100)
    # do(corpus_name='wiki-english-20171001')


