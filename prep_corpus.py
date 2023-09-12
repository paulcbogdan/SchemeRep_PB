import os.path

from gensim.utils import tokenize
from gensim.corpora import MmCorpus, Dictionary
import gensim.downloader as api
from organize_bhv import get_trial_info
from pickle_wrap import pickle_wrap
from tqdm import tqdm
import pickle
from time import time

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
    df = get_trial_info('138')
    bigrams = []
    for obj, scene in zip(df['obj_rename'], df['scene_rename']):
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

def prep_corpus_parts(n_parts=100):
    # Get list of bigrams
    # Download wiki corpus
    # Divide wiki corpus into n_parts, as loading the whole thing into memory
    #   is too intense.
    # Tokenize wiki corpus parts
    # Modify the tokens to include the bigram tokens

    fp_bigrams = r'cache/schemerep_bigrams.pkl'
    bigrams = pickle_wrap(fp_bigrams, lambda: get_bigrams(),
                          easy_override=True)
    # Input isn't tokenized yet. Downloaded corpus is a list of dictionaries,
    #   wherein the 'section_texts' key is a list of strings (not tokenized).
    # This must be converted into a list of sentences (sentence = list of tokens)
    corpus = api.load('wiki-english-20171001')
    corpus = list(corpus)
    print('Listed')
    for i in range(n_parts):
        print(f'Tokenizing part {i} of {n_parts}')
        corpus_part = corpus[i::n_parts]
        fp_part = f'cache/wiki_corpus_parts/schemerep_bigram_corpus_wiki_part{i}_{n_parts}.pkl'
        if os.path.isfile(fp_part):
            print(f'Part {i} already tokenized')
            continue
        tokenized_part = pickle_wrap(fp_part,
                 lambda: add_bigrams_to_corpus(tokenize_wiki_part(corpus_part),
                                               bigrams))

def wiki_corpus_generator(n_parts=100, parts_do=100):
    for i in range(parts_do):
        print(f'Iterating. Onto part: {i}')
        fp_part = f'cache/wiki_corpus_parts/schemerep_bigram_corpus_wiki_part{i}_{n_parts}.pkl'
        with open(fp_part, 'rb') as f:
            corpus_part = pickle.load(f)
        j = -1
        n_tokens = 0
        for j, sentence in enumerate(corpus_part):
            n_tokens += len(sentence)
            yield sentence
        print(f'\tSentences in part {i}: {j+1:,} ({n_tokens=:,})')
        del corpus_part

def prep_dictionary(corpus):
    dictionary = Dictionary(prune_at=None)
    n_iterate = 100_000
    n_done = 0
    while True:
        try:
            t = time()
            dictionary.add_documents([next(corpus) for _ in range(n_iterate)])
            n_done += n_iterate
            t_iterate = time() - t
            print(f'\tNumber done: {n_done:,} | [{t_iterate=:.3f}]')
            # if n_done % 1_000_000 == 0:
            #     dictionary.filter_extremes(no_below=1, no_above=1.0)
        except StopIteration:
            break
    return dictionary

def make_wiki_corpus_mm(parts_do=100):
    corpus = wiki_corpus_generator(parts_do=parts_do)
    t0 = time()
    print('Making dictionary...')
    fp_dict = fr'cache/wiki_dict_{parts_do}.pkl'
    dictionary = pickle_wrap(fp_dict, lambda: prep_dictionary(corpus),
                             easy_override=True)
    t1 = time()
    print(f'\t Made Dictionary in {t1 - t0:.3f} seconds')

    print('Making corpus dictionary...')
    fp_corpus_dict = fr'cache/wiki_corpus_dict_{parts_do}.pkl'
    corpus = wiki_corpus_generator(parts_do=parts_do)
    # corpus = pickle_wrap(fp_corpus_dict,
    #                      lambda: wiki_corpus_generator(parts_do=parts_do))
    # corpus_dict = [dictionary.doc2bow(text) for text in corpus]
    corpus_dict = pickle_wrap(fp_corpus_dict, 
                              lambda: [dictionary.doc2bow(text) for text in corpus],
                              easy_override=True)


    t2 = time()
    print(f'\t Made corpus dictionary in {t2 - t1:.3f} seconds')

    print('Serializing mm...')
    MmCorpus.serialize(f'cache/wiki_corpus_{parts_do}.mm', corpus_dict)
    t3 = time()
    print(f'\t Serialized mm in {t3 - t2:.3f} seconds')

# test = (i for i in range(1))
# print(next(test))
# print(next(test))
# quit()

if __name__ == '__main__':
    # get_bigram_vectors()
    make_wiki_corpus_mm()
    # prep_corpus_parts(n_parts=100)
    # do(corpus_name='wiki-english-20171001')


