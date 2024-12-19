from collections import defaultdict
from functools import cache
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.get_obj_scn_vecs import get_llama_extractor, process_cat_cat_inner
from llama.model_settings import get_explore_llama

RAM_CACHE_LLAMA_DEV = False


def fix_feature_type_classification(df):
    type2features = {}
    type2feature_cnt = {}
    for feat_type, df_feat_type in df.groupby('feature type'):
        type2features[feat_type] = df_feat_type['feature'].unique()
        type2feature_cnt[feat_type] = df_feat_type['feature'].value_counts().to_dict()

    for type0 in type2features:
        for type1 in type2features:
            if type0 >= type1:
                continue
            set0 = set(type2features[type0])
            set1 = set(type2features[type1])
            intersect = set0.intersection(set1)
            for feature in intersect:
                cnt0 = type2feature_cnt[type0][feature]
                cnt1 = type2feature_cnt[type1][feature]
                if cnt0 > cnt1:
                    df.loc[(df['feature'] == feature) & (df['feature type'] == type1), 'feature type'] = type0
                else:
                    df.loc[(df['feature'] == feature) & (df['feature type'] == type0), 'feature type'] = type1
    return df


def get_type2feat_list(df):
    type2feat_list = {}
    type2feat_map = defaultdict(dict)
    for feat_type, df_feat_type in df.groupby('feature type'):
        type2feat_list[feat_type] = sorted(df_feat_type['feature'].unique().tolist())
        for i, feat in enumerate(type2feat_list[feat_type]):
            type2feat_map[feat_type][feat] = i
    return type2feat_list, type2feat_map


def norm_pf():
    # for each feature type, normalize the pf values based on
    #   how rare a given feature is across all items.
    #   I worry this may induce negative correlations but we'll see
    pass


def get_type2RSM_(df, plot=False):
    feat2total = df.groupby('feature')['pf'].sum()
    type2feat_list, type2feat_map = get_type2feat_list(df)
    type2feat_matrix = {}
    num_items = df['concept'].nunique()
    for feat_type, feat_list in type2feat_list.items():
        type2feat_matrix[feat_type] = np.zeros((num_items, len(feat_list)))

    for i, (item, df_item) in enumerate(df.groupby('concept')):
        for feature_type, feat_list in type2feat_list.items():
            vec = [0] * len(feat_list)
            feat_map = type2feat_map[feature_type]
            for feat, pf in zip(df_item['feature'], df_item['pf']):
                if feat in feat_map:
                    vec[feat_map[feat]] = pf / feat2total[feat]
            type2feat_matrix[feature_type][i, :] = vec

    type2RSM = {}
    for feat_type, mat in type2feat_matrix.items():
        RSM = np.corrcoef(mat)
        RSM[np.diag_indices_from(RSM)] = np.nan
        type2RSM[feat_type] = RSM
        if plot:
            plt.imshow(RSM, aspect='auto', interpolation='none')
            plt.title(f'{feat_type=}')
            plt.colorbar()
            plt.show()
    return type2RSM


def get_devereux_RSM_by_type(pf_thresh=250):
    # TODO: drop any items with a parnethesis

    fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    df = pd.read_csv(fp)
    pd.set_option('display.max_rows', None)
    df = df[df['concept'].apply(lambda x: False if ('(' in x or ')' in x) else True)]
    item2pf = df.groupby('concept')['pf'].sum()
    items = item2pf[item2pf > pf_thresh].index
    df = df[df['concept'].isin(items)]
    df = fix_feature_type_classification(df)
    type2RSM = get_type2RSM_(df)
    return type2RSM

    print(df.groupby('concept')['pf'].sum().sort_values())
    quit()

    feature_types = df['feature type'].unique()
    type2RSM = {}
    items = df['concept'].unique()


    # df = fix_feature_type_classification(df)

    type2features = {}
    type2feature_cnt = {}
    for feat_type, df_feat_type in df.groupby('feature type'):
        type2features[feat_type] = df_feat_type['feature'].unique()
        type2feature_cnt[feat_type] = df_feat_type['feature'].value_counts().to_dict()

    for type0 in type2features:
        for type1 in type2features:
            if type0 >= type1:
                continue
            set0 = set(type2features[type0])
            set1 = set(type2features[type1])

            # intersect = set0.intersection(set1)
            # print(f'{type0=}, {type1=}, {intersect=}')
            assert len(set0.intersection(set1)) == 0
    quit()

    for item, df_item in df.groupby('concept'):
        pass

    for feature_type in feature_types:
        df_type = df[df['feature type'] == feature_type]

        # RSM = df_type.pivot(index='concept', columns='concept')
        # type2RSM[feature_type] = RSM
        # print(RSM)
        # quit()


def get_llama_activations_deve(item0, item1, activation_model):
    item0 = item0.replace('_', ' ')
    item1 = item1.replace('_', ' ')
    sentence = f'A {item0} and {item1}'
    t = time()
    extractor = get_llama_extractor(model_name=activation_model)
    res = extractor.extract_activations(sentence, [item0, item1], )
    print(f'Time needed for activation extraction: {time() - t:.3f} s')
    print(f'\t{[item0, item1]=} | {sentence=}')
    return res


def get_llama_d_vecs_non_normed_deve_(items, symmetric=True,
                                      cat='input', layer_name=1,
                                      activation_model='meta-llama/Llama-3.2-1b',
                                      ):
    d_vecs = {}
    cat_, inner = process_cat_cat_inner(cat)
    cnt = 0
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
                if cat == 'attn_weights':
                    v = np.nanmean(res['attn']['attn_weights'][layer_name][idx_target],
                                   axis=(0, 1))
                else:
                    v = np.nanmean(res[inner][cat_][layer_name][idx_target], axis=0)
                    if len(v.shape) > 1:
                        v = v.reshape(-1)
                if idx_target == 0:
                    # sentence f'{item 0} and {item 1}'. focus on embedding: item0
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


@cache
def get_standard_items_list(pf_thresh):
    fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    df = pd.read_csv(fp)
    pd.set_option('display.max_rows', None)
    df = df[df['concept'].apply(lambda x: False if ('(' in x or ')' in x) else True)]
    items = df['concept'].unique()
    if pf_thresh is not None:
        pf_cnt = df.groupby('concept')['pf'].sum()
        items = pf_cnt[pf_cnt > pf_thresh].index
    return items


def get_llama_d_vecs_non_normed_deve(pf_thresh=250, cat='input', layer_name=1,
                                     activation_model='meta-llama/Llama-3.2-1b',
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
    d_vecs = get_llama_d_vecs_non_normed_deve(pf_thresh, cat, layer_name, activation_model)
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


def get_deve_llama_RSM(semantic, pf_thresh=250):
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
                        activation_model='meta-llama/Llama-3.2-1b',
                        normalize=True, symmetric=True):
    d_vecs = pickle_wrap(get_llama_d_vecs_deve,
                         kwargs={'pf_thresh': pf_thresh, 'cat': cat,
                                 'layer_name': layer_name,
                                 'activation_model': activation_model,
                                 'normalize': normalize},
                         easy_override=False, verbose=-1)
    items = get_standard_items_list(pf_thresh)

    if cat != 'attn_weights':
        item2keys = defaultdict(list)
        for (order0, order1, item), v in d_vecs.items():
            item2keys[item].append((order0, order1, item))
        items_M_vecs = []
        for item in items:
            item_vecs = np.array([d_vecs[key] for key in item2keys[item]])
            M_vec = np.nanmean(item_vecs, axis=0)
            items_M_vecs.append(M_vec)
        RSM = np.corrcoef(items_M_vecs)
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


def prep_all_llama_d_vecs_deve():
    global RAM_CACHE_LLAMA_DEV
    RAM_CACHE_LLAMA_DEV = True
    llama31_3b = get_explore_llama(activation_model='meta-llama/Llama-3.2-3b',
                                   attn=False, st=0)
    llama31_3b_attn = get_explore_llama(activation_model='meta-llama/Llama-3.2-3b',
                                        attn=False, st=0)
    models = [llama31_3b, llama31_3b_attn]
    for model in models:
        get_deve_llama_RSM(model)
    get_deve_llama_RSM(models)


def do_llama_x_dev(attn=False):
    models = get_explore_llama(activation_model='meta-llama/Llama-3.2-3b',
                               attn=attn, st=0)
    RSM = get_deve_llama_RSM(models)



if __name__ == '__main__':
    # prep_all_llama_d_vecs_deve()
    get_devereux_RSM_by_type()
    # get_llama_d_vecs_deve()
    # get_llama_d_vecs_non_normed_deve()
    # fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    # df = pd.read_csv(fp)
    # pd.set_option('display.max_rows', None)
    # # print(df['feature'].value_counts())
    #
    # unique_items = df['concept'].nunique()
    # print(f'{unique_items=}')
    #
    # print(df['feature type'].value_counts())
