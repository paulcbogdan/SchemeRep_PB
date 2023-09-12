from gensim.corpora import MmCorpus
from gensim.models import Word2Vec
from pickle_wrap import pickle_wrap

from organize_bhv import get_trial_info
import pickle

def get_semantic_vectors(parts_do=25):
    fp_model = fr'cache/schemerep_word2vec_model_first_do{parts_do}.pkl'
    fp_dict = fr'cache/wiki_dict_{parts_do}.pkl'
    with open(fp_dict, 'rb') as f:
        dictionary = pickle.load(f)
    model = pickle_wrap(fp_model, lambda: fit_word2vec_mm(parts_do=parts_do),
                        easy_override=False, verbose=True)
    df = get_trial_info('138')
    d_all = {}
    for obj, scene, obj_rename, scene_rename in zip(df['obj'], df['scene'],
                          df['obj_rename'], df['scene_rename']):
        try:
            obj_rename = obj_rename.replace(' ', '_')
            scene_rename = scene_rename.replace(' ', '_')
            obj_rename = dictionary.doc2bow([obj_rename])[0]
            scene_rename = dictionary.doc2bow([scene_rename])[0]
            d_all[obj] = model.wv[obj_rename]
            d_all[scene] = model.wv[scene_rename]
        except KeyError:
            print(f'no model: {obj}')
        except IndexError:
            print(f'No dictionary: {obj}')
    return d_all

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

def fit_word2vec_mm(parts_do=5, min_count=5, epochs=5):
    import logging
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    print('Loading mm...')
    mm = MmCorpus(f'cache/wiki_corpus_{parts_do}.mm')
    print('Starting model...')
    model = Word2Vec(mm, min_count=min_count, epochs=epochs)
    return model

if __name__ == '__main__':
    PARTS_DO = 100
    fp_vecs = f'cache/schemerep_sim_vecs_{PARTS_DO}.pkl'
    d_all = pickle_wrap(fp_vecs, lambda: get_semantic_vectors(parts_do=PARTS_DO),
                        easy_override=True)