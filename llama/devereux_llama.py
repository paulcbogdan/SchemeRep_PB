import pickle
from collections import defaultdict
from functools import cache
from time import time

import numpy as np
import pandas as pd
from nltk.misc.sort import quick
from tqdm import tqdm

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.BERT_vecs import get_BERT_extractor, get_simCSE_extractor
from llama.devereux_get_quick_lists import generate_comparison_pairs_d
from llama.get_obj_scn_vecs import get_llama_extractor, process_cat_cat_inner
from llama.model_settings import get_explore_llama

# import matplotlib.pyplot as plt

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


def get_quick_items(items, quick, flip=False):
    print('Getting quick items...')
    np.random.seed(0)
    if flip:
        items0 = items[1::2]
        items1 = items[::2]
    else:
        items0 = items[::2]
        items1 = items[1::2]
    items1 = list(items1) * quick
    np.random.shuffle(items1)
    if len(items[::2]) > len(items[1::2]):
        items1 += list(np.random.choice(items1, size=quick, replace=False))
    # print(f'{len(items1)=}')
    items_mat = []
    cnt = 0
    for item0 in items0:
        items_l = []
        for i in range(quick):
            # print(f'{cnt=}')
            items_l.append(items1[cnt])
            cnt += 1
        items_mat.append(items_l)

    items_mat_d = defaultdict(list)
    for i0, item0 in enumerate(items0):
        for item1 in items_mat[i0]:
            if item0 > item1:
                items_mat_d[item1].append(item0)
            else:
                items_mat_d[item0].append(item1)
    items_mat_ = []
    for item0 in items:
        items_mat_.append(items_mat_d[item0])
    items_mat = items_mat_
    return items, items_mat



def get_llama_d_vecs_non_normed_deve_(items, symmetric=False,
                                      cat='input', layer_name=1,
                                      activation_model='meta-llama/Llama-3.2-3b-Instruct',
                                      quick=5):
    print(f'Getting llama d vecs for {cat}/{layer_name}')
    d_vecs = {}
    cat_, inner = process_cat_cat_inner(cat)
    cnt = 0
    if quick:
        print(f'Grabbing pairs for: {len(items)=}, {quick=}')
        item2item1s = generate_comparison_pairs_d(tuple(items), quick)
        print(f'\t{item2item1s[items[0]]=}')
    already_did = set()  # ???
    for i0_idx, item0 in enumerate(tqdm(items, desc='outer loop get llama')):
        if quick:
            items1 = item2item1s[item0]
        else:
            items1 = items
        for i1_idx, item1 in enumerate(items1):
            if item1 == item0:
                if quick:
                    raise ValueError(f'{item0=} ({i0_idx}) {item1=} ({i1_idx})')
                continue
            if symmetric and (quick is None):
                if item0 > item1:
                    continue
            if (item0, item1) in already_did:
                # print('Should never trigger')
                continue
            already_did.add((item0, item1))
            res, fp = pickle_wrap(get_llama_activations_deve,
                                  kwargs={'item0': item0, 'item1': item1,
                                          'activation_model': activation_model},
                                  easy_override=False, verbose=-1, dir_branches=100,
                                  RAM_cache=True, get_fp=True)
            # print(list(res['attn']))
            # if cat_ == ''


            if 'mlp_out' in res:# or 'attn_output' in res['attn']:
                # if 'mlp_out' in res:
                del res['mlp_out']
                del res['mlp_in']['down_proj']
                del res['mlp_in']['up_proj']
                del res['mlp_in']['act_fn']
                del res['attn']['q_proj']
                del res['attn']['k_proj']

                with open(fp, 'wb') as f:
                    pickle.dump(res, f)

            # clean up the res dict. BERT/simCSE won't have dicts here
            if isinstance(res, dict):
                for inner_, d in res.items():
                    if inner_ in ['mlp_out', 'mlp_in', 'attn']:
                        del_keys = []
                        for outer in d.keys():
                            if outer != cat_:
                                del_keys.append(outer)
                        for outer in del_keys:
                            del res[inner_][outer]

            for idx_target in [0, 1]:
                if activation_model in ['BERT', 'simCSE']:
                    v = res[idx_target][layer_name]
                elif cat in 'attn_weights':
                    v = np.nanmean(res['attn']['attn_weights'][layer_name][idx_target],
                                   axis=(0, 1))
                else:
                    v = np.nanmean(res[inner][cat_][layer_name][idx_target], axis=0)
                    if len(v.shape) > 1:
                        v = v.reshape(-1)
                if idx_target == 0:  # sentence f'{item 0} and {item 1}'. focus on embedding: item0
                    d_vecs[(item0, item1, item0)] = v
                    if symmetric:
                        d_vecs[(item1, item0, item0)] = v
                else:
                    d_vecs[(item0, item1, item1)] = v
                    if symmetric:
                        d_vecs[(item1, item0, item1)] = v
            cnt += 1
            if cnt % 100 == 0:
                if quick:
                    total = len(items) * quick
                else:
                    total = len(items) * (len(items) - 1)
                    total = total // 2 if symmetric else total
                print(f'{cnt=} / {total=}')
    return d_vecs


@cache
def get_standard_items_list(pf_thresh, item_standard='deve', in_both=True,
                            norm_per_concept=True):
    if item_standard == 'deve':
        fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    elif item_standard == 'mariam':
        fp = r'C:\PycharmProjects\SchemeRep\llama\features\Mariam_norm_dict_matched.csv'
    else:
        raise ValueError(f'Invalid item_standard: {item_standard}')

    if in_both and not pf_thresh > 315:
        fp_deve = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
        df_deve = pd.read_csv(fp_deve)
        df_deve = df_deve[df_deve['concept'].apply(lambda x: False if ('(' in x or ')' in x) else True)]
        df_deve['feature type'] = df_deve['feature type'].apply(
            lambda x: 'encyclopedic' if x == 'encyclopaedic' else x)
        df_deve['which'] = 'deve'
        fp_mariam = r'C:\PycharmProjects\SchemeRep\llama\features\Mariam_norm_dict_matched.csv'
        df_mariam = pd.read_csv(fp_mariam)
        df_deve = df_deve[df_deve['in_both']]
        deve_pf_per_concept = df_deve['pf'].sum() / len(df_deve['concept'].unique())
        if norm_per_concept:
            df_deve['pf'] = df_deve['pf'] / deve_pf_per_concept

        df_mariam = df_mariam[df_mariam['in_both']]
        df_mariam['which'] = 'mariam'
        mariam_pf_per_concept = df_mariam['pf'].sum() / len(df_mariam['concept'].unique())
        if norm_per_concept:
            df_mariam['pf'] = df_mariam['pf'] / mariam_pf_per_concept
        # dividing as so makes it so a larger participant sample in deve/mariam won't bias
        df = pd.concat([df_deve, df_mariam])
        df['feature'] = df['feature'].apply(lambda x: x.replace(' ', '_'))

    else:
        df = pd.read_csv(fp)
        pd.set_option('display.max_rows', None)
        df = df[df['concept'].apply(lambda x: False if ('(' in x or ')' in x) else True)]
        df['feature'] = df['feature'].apply(lambda x: x.replace(' ', '_'))

    if pf_thresh > len(df['concept'].unique()):
        pass
    elif pf_thresh is not None :
        pf_cnt = df.groupby('concept')['pf'].sum()
        pf_cnt = pf_cnt.sort_values(ascending=False)
        pf_thresh = pf_cnt.iloc[pf_thresh] - .0000001
        items = pf_cnt[pf_cnt > pf_thresh].index
        items = sorted(items.to_list())
        if len(items) % 2 == 1:
            items = items[:-1]
        df = df[df['concept'].isin(items)]

    items = df['concept'].unique()
    return items, df



def get_llama_d_vecs_non_normed_deve(pf_thresh=250, cat='input', layer_name=1,
                                     activation_model='meta-llama/Llama-3.2-3b',
                                     quick=5, item_standard='deve'):
    items, _ = get_standard_items_list(pf_thresh, item_standard=item_standard)
    d_vecs, fp = pickle_wrap(get_llama_d_vecs_non_normed_deve_,
                         kwargs={'items': items, 'symmetric': False,
                                 'cat': cat, 'layer_name': layer_name,
                                 'activation_model': activation_model,
                                 'quick': quick,},
                         easy_override=False, verbose=-1,
                         get_fp=True)
    return d_vecs


def get_llama_d_vecs_deve(pf_thresh=250, cat='input', layer_name=1,
                          activation_model='meta-llama/Llama-3.2-1b',
                          normalize=True, quick=5, item_standard='deve'):
    d_vecs = get_llama_d_vecs_non_normed_deve(pf_thresh, cat, layer_name,
                                              activation_model,
                                              quick, item_standard)
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


def get_deve_llama_RSM_l(semantic_l, pf_thresh=250, quick=5, item_standard='deve'):
    # if semantic_l[-1] == 'delta':
    #     RSM0 = pickle_wrap(get_deve_llama_RSM,
    #                        kwargs={'semantic': semantic_l[0], 'pf_thresh': pf_thresh,
    #                                'quick': quick, 'item_standard': item_standard},
    #                        easy_override=False, verbose=-1)
    #     RSM1 = pickle_wrap(get_deve_llama_RSM,
    #                        kwargs={'semantic': semantic_l[1], 'pf_thresh': pf_thresh,
    #                                   'quick': quick, 'item_standard': item_standard},
    #                        easy_override=False, verbose=-1)
    #     RSM = RSM0 - RSM1
    # else:
    all_RSM = []
    for semantic in semantic_l:
        RSM = pickle_wrap(get_deve_llama_RSM,
                          kwargs={'semantic': semantic, 'pf_thresh': pf_thresh,
                                  'quick': quick, 'item_standard': item_standard},
                          easy_override=True, verbose=-1)
        all_RSM.append(RSM)
    RSM = np.nanmean(all_RSM, axis=0)
    return RSM


def get_deve_llama_RSM(semantic, pf_thresh=100,
                       quick=5, item_standard='deve',
                       position=None
                       ):

    if isinstance(semantic, list):
        RSM = pickle_wrap(get_deve_llama_RSM_l,
                          kwargs={'semantic_l': semantic, 'pf_thresh': pf_thresh,
                                  'quick': quick, 'item_standard': item_standard},
                          easy_override=False, verbose=1)
    elif isinstance(semantic, tuple):
        cat = semantic[1]
        layer_name = semantic[2]
        activation_model = semantic[4]
        normalize = semantic[5]
        RSM = pickle_wrap(get_deve_llama_RSM_,
                          kwargs={'pf_thresh': pf_thresh, 'cat': cat,
                                  'layer_name': layer_name,
                                  'activation_model': activation_model,
                                  'normalize': normalize,
                                  'quick': quick,
                                  'item_standard': item_standard,
                                  'position': position
                                  # 'only_second': only_second
                                  },
                          easy_override=False, verbose=1)
    else:
        raise ValueError
    return RSM

def get_item_prod_sum(items, cat, items_M_vecs):
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
    return vecs

def get_llama_vecs_ar(pf_thresh=250, cat='input', layer_name=1,
                      activation_model='meta-llama/Llama-3.2-3b',
                      normalize=True, quick=5, item_standard='deve',
                      position=None):
    # if position is not None:
    #     assert quick is None
    cat_ = cat if isinstance(cat, str) else cat[0]
    d_vecs = pickle_wrap(get_llama_d_vecs_deve,
                         kwargs={'pf_thresh': pf_thresh, 'cat': cat_,
                                 'layer_name': layer_name,
                                 'activation_model': activation_model,
                                 'normalize': normalize,
                                 'quick': quick,
                                 'item_standard': item_standard},
                         easy_override=False, verbose=-1)
    print(f'\tGathered d_vecs ({quick=})')
    items, _ = get_standard_items_list(pf_thresh)
    print(f'\t{len(items)=}')
    if cat != 'attn_weights' or True:
        item2keys = defaultdict(list)
        for (order0, order1, item), v in d_vecs.items():
            if position is None:
                item2keys[item].append((order0, order1, item))
            elif position == 0:
                if order0 == item:
                    item2keys[item].append((order0, order1, item))
            elif position == 1:
                if order1 == item:
                    item2keys[item].append((order0, order1, item))
            elif position == 'flip':
                if order1 == item:
                    item2keys[order0].append((order0, order1, item))
            elif position == 'flip_sanity':
                if order0 == item:
                    item2keys[order1].append((order0, order1, item))
            else:
                raise ValueError

        items_M_vecs = []
        for item in items:
            item_vecs = np.array([d_vecs[key] for key in item2keys[item]])
            item_vecs[np.isinf(item_vecs)] = np.nan
            M_vec = np.nanmean(item_vecs, axis=0)
            items_M_vecs.append(M_vec)
        items_M_vecs = np.array(items_M_vecs)
        items_M_vecs_ = []
        for v in items_M_vecs:
            if np.any(np.isnan(v)):
                M = np.nanmean(v)
                v = np.nan_to_num(v, nan=M)
            items_M_vecs_.append(v)
        items_M_vecs = np.array(items_M_vecs_)
        if isinstance(cat, tuple):
            assert cat[1] in ['item_sum', 'item_prod']
            vecs = get_item_prod_sum(items, cat, items_M_vecs)
        else:
            vecs = items_M_vecs
    else:
        raise ValueError
        assert len(items) < 300, 'Too many items for attn_weights. RAM danger maybe'
        vecs = []
        for item0 in items:
            for item1 in items:
                if item0 >= item1:
                    continue
                v = (d_vecs[(item0, item1, item0)] + d_vecs[(item0, item1, item1)]) / 2
                vecs.append(v)
    return vecs

def get_deve_llama_RSM_(pf_thresh=250, cat='input', layer_name=1,
                        activation_model='meta-llama/Llama-3.2-3b',
                        normalize=True,
                        quick=5, item_standard='deve',
                        position=None,
                        # only_second=True
                        ):
    if isinstance(quick, tuple):
        quick0 = quick[0]
        quick1 = quick[1]
        vecs0 = get_llama_vecs_ar(pf_thresh, cat, layer_name,
                                  activation_model, normalize, quick0, item_standard,
                                  position)
        vecs1 = get_llama_vecs_ar(pf_thresh, cat, layer_name,
                                  activation_model, normalize, quick1, item_standard,
                                  position)
        vecs = vecs0 - vecs1
    else:
        vecs = get_llama_vecs_ar(pf_thresh, cat, layer_name,
                                 activation_model, normalize, quick, item_standard,
                                 position)
    RSM = np.corrcoef(vecs)
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
                               # activation_model='meta-llama/Llama-3.3-70b-Instruct',
                               pf_thresh=300, quick=None, item_standard='deve'):
    global RAM_CACHE_LLAMA_DEV
    RAM_CACHE_LLAMA_DEV = True
    # llama31_3b = get_explore_llama(activation_model=activation_model,
    #                                attn='attn_output', st=0)
    llama31_3b = get_explore_llama(activation_model=activation_model,
                                   attn=False, st=0)
    models = llama31_3b

    # TODO: attn_outpuut
    # llama31_3b_output = get_explore_llama(activation_model=activation_model,
    #                                       attn='attn_output', st=0)
    # models += llama31_3b_output
    #
    # llama31_3b = get_explore_llama(activation_model=activation_model,
    #                                attn=False, st=0)
    # models += llama31_3b

    # models = get_dev_explore_BERT('BERT')
    # models = models + models_
    # models = get_dev_explore_BERT('simCSE')
    # models = models + models_
    # models = models + models_simCSE
    # llama31_3b_attn = get_explore_llama(activation_model=activation_model,
    #                                     attn=True, st=0)
    # models = [llama31_3b, llama31_3b_attn]
    for model in models:
        get_deve_llama_RSM(model, pf_thresh=pf_thresh,
                           quick=quick, item_standard=item_standard)
    # get_deve_llama_RSM(models, pf_thresh=pf_thresh,
    #                    quick=quick, item_standard=item_standard)


if __name__ == '__main__':
    prep_all_llama_d_vecs_deve(quick=1)

    # prep_all_llama_d_vecs_deve()
    # prep_all_llama_d_vecs_deve(quick=4)
    # prep_all_llama_d_vecs_deve(quick=50)

    # TODO: Try Llama-3.2-1b

    # TODO: turn off symmetry
