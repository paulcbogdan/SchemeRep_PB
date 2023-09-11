from gensim.corpora import MmCorpus
from gensim.models import Word2Vec
from pickle_wrap import pickle_wrap

from organize_bhv import organize_subj_df
import pickle

def get_vectors(parts_do=25):
    fp_model = fr'cache/schemerep_word2vec_model_first_do{parts_do}.pkl'
    # model = pickle_wrap(fp_model, lambda: fit_word2vec(parts_do=parts_do),
    #                     easy_override=False, verbose=True)

    fp_dict = fr'cache/wiki_dict_{parts_do}.pkl'
    with open(fp_dict, 'rb') as f:
        dictionary = pickle.load(f)
    dictionary.filter_extremes(no_below=2, no_above=0.5)

    model = pickle_wrap(fp_model, lambda: fit_word2vec_mm(parts_do=parts_do),
                        easy_override=True, verbose=True)
    # print(model.wv[dictionary.doc2bow(['the'])])
    # quit()
    df = organize_subj_df('138')
    d_obj = {}
    d_scene = {}
    d_all = {}
    for obj, scene in zip(df['obj'], df['scene']):
        print(f'1: {obj}')
        try:
            obj = obj.replace(' ', '_')
            # print(obj)
            obj = dictionary.doc2bow([obj])[0]
            # print(obj)
            # scene = scene.replace(' ', '_')
            d_all[obj] = d_obj[obj] = model.wv[obj]
            # d_all[scene] = d_scene[scene] = model.wv[scene]
        except KeyError:
            print(f'no model: {obj}')
            # pass
        except IndexError:
            print(f'No dictionary: {obj}')
    return d_all
    #
    # df_as_l = []
    # for key, vec in d_obj.items():
    #     d = {'stim': key, 'vec': vec, 'type': 'obj'}
    #     df_as_l.append(d)
    # for key, vec in d_scene.items():
    #     d = {'stim': key, 'vec': vec, 'type': 'scene'}
    #     df_as_l.append(d)
    #
    # return d_obj, d_scene, df_as_l




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
    d_obj, d_scene, df_as_l = pickle_wrap(fp_vecs,
                                          lambda: get_vectors(parts_do=PARTS_DO),
                                          easy_override=True)