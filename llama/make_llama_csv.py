from operator import index

import numpy as np

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_llama import get_llama_d_vecs_non_normed_deve_
from llama.get_obj_scn_vecs import get_llama_d_vecs
from llama.model_settings import get_explore_llama
from organize_bhv import get_trial_info
import pandas as pd
from pathlib import Path

def get_llama_df(semantic_tup):
    activation_model = semantic_tup[4]
    if semantic_tup[3] == 'obj_solo':
        activation_model = (activation_model, 'obj_solo')
    normalize = semantic_tup[5]
    # obj_scn_norm = semantic_tup[5]
    df_sn = get_trial_info(102, easy_override=False, verbose=-1)
    objs = df_sn['obj'].unique().tolist()

    cat_ = semantic_tup[1]
    all_possible = True if semantic_tup[3] in ['obj_M', 'obj_dif',
                                               'scn_M'] else False

    d_vecs = pickle_wrap(get_llama_d_vecs, kwargs={'cat': cat_,
                                                   'layer_name': semantic_tup[2],
                                                   'normalize': normalize,
                                                   'activation_model': activation_model,
                                                   'all_possible': all_possible,
                                                   },
                         easy_override=False, verbose=-1,
                         RAM_cache=True)

    vecs = np.array([d_vecs[(None, obj)] for obj in objs])
    columns = [f'dim_{i}' for i in range(vecs.shape[1])]
    df = pd.DataFrame(vecs, index=objs, columns=columns)
    return df


def get_llama_dino_df(activation_model='meta-llama/Llama-3.2-3b',
                      cat='input', layer_name=0, normalize=True,
                      do_pair=True):
    if do_pair:
        fp = r'C:\PycharmProjects\SchemeRep\llama\features\DinoWordPairs_01.17.2025.csv'
        df_words = pd.read_csv(fp)
        df_words = df_words.loc[~pd.isna(df_words['Object1DisplayName'])]
        words0 = df_words['Object1DisplayName'].to_list()
        words1 = df_words['Object2DisplayName'].to_list()
        words = (words0, words1)
    else:
        fp = r'C:\PycharmProjects\SchemeRep\llama\features\DinoWords_01.17.2025.csv'
        df_words = pd.read_csv(fp)
        words = df_words['Object1DisplayName'].to_list()


    d_vecs, fp = pickle_wrap(get_llama_d_vecs_non_normed_deve_,
                             kwargs={'items': words, 'symmetric': False,
                                     'cat': cat, 'layer_name': layer_name,
                                     'activation_model': activation_model,
                                     'quick': 1, },
                             easy_override=False, verbose=-1,
                             get_fp=True)
    print(list(d_vecs.keys()))
    if isinstance(normalize, bool) and normalize:
        vecs_ar = np.array([v for v in d_vecs.values()])
        M = np.nanmean(vecs_ar, axis=0)
        SD = np.nanstd(vecs_ar, axis=0)
        d_vecs = {k: (v - M) / SD for k, v in d_vecs.items()}
    elif not isinstance(normalize, bool):
        raise ValueError
    else:
        pass
    if do_pair:
        vecs = []
        for word0, word1 in zip(words[0], words[1]):
            v0 = d_vecs[(word0, word1, word1)]
            v1 = d_vecs[(word1, word0, word0)]
            vec = (v0 + v1) / 2
            vecs.append(vec)
        vecs = np.array(vecs)
        columns = [f'dim_{i}' for i in range(vecs.shape[1])]
        df = pd.DataFrame(vecs, columns=columns)
        df.insert(0, 'Object2DisplayName', df_words['Object2DisplayName'].to_list())
        df.insert(0, 'ID2', df_words['ID2'].to_list())
        df.insert(0, 'Object1DisplayName', df_words['Object1DisplayName'].to_list())
        df.insert(0, 'ID1', df_words['ID1'].to_list())
        df.reset_index(drop=True, inplace=True)
        print(df)
        # print(len(columns))
        # quit()
    else:

        word2tup = {}
        for tup in d_vecs.keys():
            word0 = tup[0]
            if word0 == tup[2]:
                word2tup[word0] = tup

        vecs = []
        for word in words:
            vecs.append(d_vecs[word2tup[word]])
        vecs = np.array(vecs)
        columns = [f'dim_{i}' for i in range(vecs.shape[1])]
        df = pd.DataFrame(vecs, index=words, columns=columns)
        df.insert(0, 'ID', df_words['ID'].to_list())
        df.insert(0, 'Object1DisplayName', words)
        df.reset_index(drop=True, inplace=True)
        # print(len(columns))
        # quit()

    return df

def make_all_dinolab(cat='input', do_pair=True):
    activation_model = 'meta-llama/Llama-3.2-3b'
    model2name = {'meta-llama/Llama-3.2-3b': 'Llama-3.2-3b',}
    name = model2name[activation_model]

    for layer_name in range(28):
        df = get_llama_dino_df(activation_model='meta-llama/Llama-3.2-3b',
                            cat=cat, layer_name=layer_name, normalize=True,
                               do_pair=do_pair)

        if do_pair:
            fn = f'pair_vecs_{name}_{cat}_{layer_name}.csv'
        else:
            fn = f'vecs_{name}_{cat}_{layer_name}.csv'
        fp = fr'llama/vec_csvs_DinoLab/{name}/{cat}/{fn}'
        Path(fp).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(fp, index=False)
        print(f'Made: {fp=}')


def make_all_schemerep_csv():
    ACTIVATION_MODEL = 'meta-llama/Llama-3.2-3b'
    MODEL2NAME = {'meta-llama/Llama-3.2-3b': 'Llama-3.2-3b',}
    SEMANTIC_L = get_explore_llama(activation_model=ACTIVATION_MODEL,
                                   attn=False, do_M='obj_solo', last_only=False,
                                   include_scn=False, normalize=True
                                   )
    for SEMANTIC in SEMANTIC_L:
        name = MODEL2NAME[ACTIVATION_MODEL]
        cat = SEMANTIC[1]
        layer = SEMANTIC[2]

        fn = f'vecs_{name}_{cat}_{layer}.csv'
        fp = fr'llama/vec_csvs/{name}/{cat}/{fn}'
        Path(fp).parent.mkdir(parents=True, exist_ok=True)

        df = get_llama_df(SEMANTIC)
        df.to_csv(fp)
        print(f'Made: {fp=}')


if __name__ == '__main__':
    make_all_dinolab()


