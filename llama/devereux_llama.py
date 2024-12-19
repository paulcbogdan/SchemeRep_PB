from collections import defaultdict
from functools import cache
from time import time

import numpy as np
import pandas as pd

from Utils.pickle_wrap_funcs import pickle_wrap

from llama.BERT_vecs import get_BERT_extractor, get_simCSE_extractor
from llama.get_obj_scn_vecs import get_llama_extractor, process_cat_cat_inner
from llama.model_settings import get_explore_llama

RAM_CACHE_LLAMA_DEV = False
ITEM_STANDARD = 'deve'
ITEM_STANDARD = 'mariam'

def get_llama_activations_deve(item0, item1, activation_model):
    item0 = item0.replace('_', ' ')
    item1 = item1.replace('_', ' ')
    if item0[0].lower() in ['a', 'e', 'i', 'o', 'u']:
        sentence = f'An {item0} and {item1}'
    else:
        sentence = f'A {item0} and {item1}'
    t = time()
    if activation_model == 'BERT':
        extractor = get_BERT_extractor()
    elif activation_model == 'simCSE':
        extractor = get_simCSE_extractor()
    else:
        extractor = get_llama_extractor(model_name=activation_model)
    res = extractor.extract_activations(sentence, [item0, item1], )
    print(f'Time needed for activation extraction: {time() - t:.3f} s')
    print(f'\t{[item0, item1]=} | {sentence=}')
    return res


def get_llama_d_vecs_non_normed_deve_(items, symmetric=True,
                                      cat='input', layer_name=1,
                                      activation_model='meta-llama/Llama-3.2-1b',
                                      ):
    print(f'Getting llama d vecs for {cat}/{layer_name}')
    d_vecs = {}
    cat_, inner = process_cat_cat_inner(cat)
    cnt = 0
    # print(f'{len(items)=}')
    # quit()
    # items = items[::-1]
    for i0_idx, item0 in enumerate(items):
        for i1_idx, item1 in enumerate(items):
            if item1 == item0:
                continue
            if symmetric:
                if item0 > item1:
                    continue
            res = pickle_wrap(get_llama_activations_deve,
                              kwargs={'item0': item0, 'item1': item1,
                                      'activation_model': activation_model},
                              easy_override=False, verbose=-1, dir_branches=100,
                              RAM_cache=RAM_CACHE_LLAMA_DEV)
            for idx_target in [0, 1]:
                if activation_model in ['BERT', 'simCSE']:
                    v = res[idx_target][layer_name]
                elif cat == 'attn_weights':
                    v = np.nanmean(res['attn']['attn_weights'][layer_name][idx_target],
                                   axis=(0, 1))
                else:
                    v = np.nanmean(res[inner][cat_][layer_name][idx_target], axis=0)
                    if len(v.shape) > 1:
                        v = v.reshape(-1)
                if idx_target == 0: # sentence f'{item 0} and {item 1}'. focus on embedding: item0
                    d_vecs[(item0, item1, item0)] = v
                    if symmetric:
                        d_vecs[(item1, item0, item0)] = v
                else:
                    d_vecs[(item0, item1, item1)] = v
                    if symmetric:
                        d_vecs[(item1, item0, item1)] = v
            cnt += 1
            if cnt % 100 == 0:
                total = len(items) * (len(items) - 1)
                total = total // 2 if symmetric else total
                print(f'{cnt=} / {total=}')
    return d_vecs


# def get_deve_df_threshed(pf_thresh, in_both=True):
#     if ITEM_STANDARD == 'deve':
#         fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
#     elif ITEM_STANDARD == 'mariam':
#         fp = r'C:\PycharmProjects\SchemeRep\llama\features\Mariam_norm_dict.csv'
#     else:
#         raise ValueError(f'Invalid ITEM_STANDARD: {ITEM_STANDARD=}')
#     df = pd.read_csv(fp)
#     pd.set_option('display.max_rows', None)
#     df = df[df['concept'].apply(lambda x: False if ('(' in x or ')' in x) else True)]
#     if in_both:
#         df = df[df['in_both']]
#     items = df['concept'].unique()
#     if pf_thresh is not None:
#         pf_cnt = df.groupby('concept')['pf'].sum()
#         pf_cnt = pf_cnt.sort_values(ascending=False)
#         pf_thresh = pf_cnt.iloc[pf_thresh] - .01
#         df = df[df['pf'] > pf_thresh]
#     return df

@cache
def get_standard_items_list(pf_thresh, item_standard='deve', in_both=True):
    if item_standard == 'deve':
        fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    elif item_standard == 'mariam':
        fp = r'C:\PycharmProjects\SchemeRep\llama\features\Mariam_norm_dict.csv'
    else:
        raise ValueError(f'Invalid item_standard: {item_standard}')
    df = pd.read_csv(fp)
    pd.set_option('display.max_rows', None)
    df = df[df['concept'].apply(lambda x: False if ('(' in x or ')' in x) else True)]
    if in_both:
        df = df[df['in_both']]
    items = df['concept'].unique()

    if pf_thresh is not None:
        pf_cnt = df.groupby('concept')['pf'].sum()
        pf_cnt = pf_cnt.sort_values(ascending=False)
        pf_thresh = pf_cnt.iloc[pf_thresh] - .01
        items = pf_cnt[pf_cnt > pf_thresh].index
        items = sorted(items.to_list())
        df = df[df['concept'].isin(items)]
    return items, df


def get_llama_d_vecs_non_normed_deve(pf_thresh=250, cat='input', layer_name=1,
                                     activation_model='meta-llama/Llama-3.2-3b',
                                     ):
    items = get_standard_items_list(pf_thresh)
    d_vecs = pickle_wrap(get_llama_d_vecs_non_normed_deve_,
                         kwargs={'items': items, 'symmetric': True,
                                 'cat': cat, 'layer_name': layer_name,
                                 'activation_model': activation_model},
                         easy_override=False, verbose=-1)
    return d_vecs


def get_llama_d_vecs_deve(pf_thresh=250, cat='input', layer_name=1,
                          activation_model='meta-llama/Llama-3.2-1b',
                          normalize=True):
    d_vecs = get_llama_d_vecs_non_normed_deve(pf_thresh, cat, layer_name,
                                              activation_model)
    if isinstance(normalize, bool) and normalize:
        vecs_ar = np.array([v for v in d_vecs.values()])
        M = np.nanmean(vecs_ar, axis=0)
        SD = np.nanstd(vecs_ar, axis=0)
        d_vecs = {k: (v - M) / SD for k, v in d_vecs.items()}
    elif not isinstance(normalize, bool):
        raise ValueError
    else:
        pass
    return d_vecs


def get_deve_llama_RSM_l(semantic_l, pf_thresh=250):
    all_RSM = []
    for semantic in semantic_l:
        RSM = pickle_wrap(get_deve_llama_RSM,
                          kwargs={'semantic': semantic, 'pf_thresh': pf_thresh},
                          easy_override=False, verbose=-1)
        all_RSM.append(RSM)
    RSM = np.nanmean(all_RSM, axis=0)
    return RSM


def get_deve_llama_RSM(semantic, pf_thresh=100):
    if isinstance(semantic, list):
        RSM = pickle_wrap(get_deve_llama_RSM_l,
                          kwargs={'semantic_l': semantic, 'pf_thresh': pf_thresh},
                          easy_override=False, verbose=-1)
    elif isinstance(semantic, tuple):
        cat = semantic[1]
        layer_name = semantic[2]
        activation_model = semantic[4]
        normalize = semantic[5]
        RSM = pickle_wrap(get_deve_llama_RSM_,
                          kwargs={'pf_thresh': pf_thresh, 'cat': cat,
                                  'layer_name': layer_name,
                                  'activation_model': activation_model,
                                  'normalize': normalize},
                          easy_override=False, verbose=-1)
    else:
        raise ValueError
    return RSM


def get_deve_llama_RSM_(pf_thresh=250, cat='input', layer_name=1,
                        activation_model='meta-llama/Llama-3.2-3b',
                        normalize=True, symmetric=True):
    print('Making dev-llama RSM')
    cat_ = cat if isinstance(cat, str) else cat[0]
    d_vecs = pickle_wrap(get_llama_d_vecs_deve,
                         kwargs={'pf_thresh': pf_thresh, 'cat': cat_,
                                 'layer_name': layer_name,
                                 'activation_model': activation_model,
                                 'normalize': normalize},
                         easy_override=False, verbose=-1)
    print(f'\tGathered d_vecs')
    items = get_standard_items_list(pf_thresh)
    if cat != 'attn_weights':
        item2keys = defaultdict(list)
        for (order0, order1, item), v in d_vecs.items():
            item2keys[item].append((order0, order1, item))
        items_M_vecs = []
        for item in items:
            item_vecs = np.array([d_vecs[key] for key in item2keys[item]])
            item_vecs[np.isinf(item_vecs)] = np.nan
            M_vec = np.nanmean(item_vecs, axis=0)
            items_M_vecs.append(M_vec)
        bad_cols = np.isnan(items_M_vecs).all(axis=0)
        items_M_vecs = np.array(items_M_vecs)[:, ~bad_cols]
        items_M_vecs_ = []
        for v in items_M_vecs:
            if np.any(np.isnan(v)):
                M = np.nanmean(v)
                v = np.nan_to_num(v, nan=M)
            items_M_vecs_.append(v)
        items_M_vecs = np.array(items_M_vecs_)
        if isinstance(cat, tuple):
            assert cat[1] in ['item_sum', 'item_prod']
            vecs = []
            for i, item0 in enumerate(items):
                for j, item1 in enumerate(items):
                    if item0 >= item1: continue
                    if cat[1] == 'item_sum':
                        vecs.append(items_M_vecs[i] + items_M_vecs[j])
                    elif cat[1] == 'item_prod':
                        vecs.append(items_M_vecs[i] * items_M_vecs[j])
                    else:
                        raise ValueError
            RSM = np.corrcoef(vecs)
        else:
            RSM = np.corrcoef(items_M_vecs)
        RSM[np.diag_indices_from(RSM)] = np.nan
    else:
        assert len(items) < 300
        pair_item_vs = []
        for item0 in items:
            for item1 in items:
                if item0 >= item1:
                    continue
                v = (d_vecs[(item0, item1, item0)] + d_vecs[(item0, item1, item1)]) / 2
                pair_item_vs.append(v)
        RSM = np.corrcoef(pair_item_vs)

    return RSM

def get_dev_explore_BERT(bert_type='BERT', st=0):
    layers = list(range(st, 13))
    semantic_l = []
    normalize = True
    for layer in layers:
        semantic = ('BERT', bert_type, layer, 'obj', bert_type, normalize)
        semantic_l.append(semantic)
    return semantic_l

def prep_all_llama_d_vecs_deve(activation_model='meta-llama/Llama-3.2-3b',
                               pf_thresh=100):
    global RAM_CACHE_LLAMA_DEV
    RAM_CACHE_LLAMA_DEV = True
    # llama31_3b = get_explore_llama(activation_model=activation_model,
    #                                attn='v_proj', st=0)
    llama31_3b = get_explore_llama(activation_model=activation_model,
                                   attn=False, st=0)
    models = llama31_3b

    # models = get_dev_explore_BERT('BERT')
    # models_simCSE = get_dev_explore_BERT('simCSE')
    # models = models + models_simCSE
    # llama31_3b_attn = get_explore_llama(activation_model=activation_model,
    #                                     attn=True, st=0)
    # models = [llama31_3b, llama31_3b_attn]
    for model in models:
        get_deve_llama_RSM(model, pf_thresh=pf_thresh)
    get_deve_llama_RSM(models, pf_thresh=pf_thresh)


if __name__ == '__main__':
    prep_all_llama_d_vecs_deve()