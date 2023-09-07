from gensim.models import Phrases
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import gensim.downloader as api
from test2 import organize_subj_df

import numpy as np
import pandas as pd
import string
import gensim
from gensim.models import phrases, word2vec


# bigrams = phrases.Phrases(sentences)

def add_bigrams_to_corpus(corpus, bigrams):
    for bigram in bigrams:
        corpus_new = []
        bigram_in = False
        for sentence in corpus:
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

def prep_bigram_corpus(corpus_name='text8'):
    bigrams = get_bigrams()
    corpus = api.load('text8')
    corpus = add_bigrams_to_corpus(corpus, bigrams)
    quit()

print(api.load())
quit()
# print(bigrams)
# quit()
#
# corpus = [['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against', 'early', 'as', 'a'],
#           ['as', 'a', 'as', 'stuff']]
# bigrams_ = [('as', 'a'), ('term', 'of', 'abuse')]



# print(type(corpus))
# print(len([x for x in corpus]))
for x in corpus:
    print(x)
    break
model = Word2Vec(min_count=1)
print('toast')
model.build_vocab(corpus)
test = model.train(corpus, total_examples=1, epochs=1)
print(test)
print(model.wv['abusssdadase'])

