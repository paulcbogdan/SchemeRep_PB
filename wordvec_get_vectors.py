from gensim.corpora import MmCorpus
from gensim.models import Word2Vec
from pickle_wrap import pickle_wrap

from organize_bhv import get_trial_info
import pickle
import gensim
import gensim.downloader
import numpy as np

def get_semantic_vectors_OLD_(parts_do=25):
    fp_model = fr'cache/schemerep_word2vec_model_first_do{parts_do}.pkl'
    fp_dict = fr'cache/wiki_dict_{parts_do}.pkl'
    with open(fp_dict, 'rb') as f:
        dictionary = pickle.load(f)

    # with open(fp_model, 'rb') as file:
        # model = pickle.load(file)
    model = pickle_wrap(fp_model, lambda: fit_word2vec_mm(parts_do=parts_do),
                        easy_override=False, verbose=True)
    df = get_trial_info('138')
    d_all = {}
    for obj, scene, obj_rename, scene_rename in zip(df['obj'], df['scene'],
                          df['obj_rename'], df['scene_rename']):
        try:
            obj_rename = obj_rename.replace(' ', '_')
            scene_rename = scene_rename.replace(' ', '_')
            obj_rename = dictionary.doc2bow([obj_rename])[0][0]
            scene_rename = dictionary.doc2bow([scene_rename])[0][0]
            d_all[obj] = model.wv[obj_rename]
            d_all[scene] = model.wv[scene_rename]
        except KeyError:
            print(f'no model: {obj}')
        except IndexError:
            print(f'No dictionary: {obj}')
    return d_all

def get_semantic_vectors_OLD(parts_do=100):
    # fit using python 3.11
    fp_vecs = f'cache/schemerep_sem_vecs_{parts_do}.pkl'
    d_vecs = pickle_wrap(fp_vecs, lambda: get_semantic_vectors_OLD_(parts_do),
                         easy_override=False)
    return d_vecs

def do_vec(stim, w2v):
    parts = stim.split(' ')
    vecs = []
    for part in parts:
        try:
            vec = w2v[part]
            vecs.append(vec)
        except KeyError:
            print(f'Bad {stim}: {part}')
    return np.mean(vecs, axis=0)

def get_semantic_vectors_():
    w2vectors = gensim.downloader.load('word2vec-google-news-300')
    print('Loaded word2vec')
    df = get_trial_info('138')
    d_all = {}
    for obj, scene, obj_rename, scene_rename in zip(df['obj'], df['scene'],
                          df['obj_rename'], df['scene_rename']):
        d_all[obj] = do_vec(obj_rename, w2vectors)
        d_all[scene] = do_vec(scene_rename, w2vectors)
    return d_all

def get_semantic_vectors():
    # fit using python 3.11
    fp_vecs = f'cache/schemerep_sem_vecs.pkl'
    d_vecs = pickle_wrap(fp_vecs, get_semantic_vectors_,
                         easy_override=False)
    return d_vecs


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
    d_all = get_semantic_vectors()
    vec = d_all[next(iter(d_all))]
    print(f'{vec.shape=}')