import pandas as pd

from functools import cache
from time import time

from Utils.pickle_wrap_funcs import pickle_wrap
from org_sns import get_sns
from organize_bhv import get_trial_info
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

@cache
def get_llama_extractor(model_name='meta-llama/Llama-3.2-1b'):
    from llama.llama_test import LlamaActivationExtractor
    extractor = LlamaActivationExtractor(model_name)
    extractor._register_comprehensive_hooks()
    return extractor

def get_obj2grammar():
    df = pd.read_csv(r'C:\PycharmProjects\SchemeRep\llama\obj_w_grammar.csv')
    obj2grammar = {obj: grammar for obj, grammar in zip(df['obj'], df['grammar'])}
    obj2override = {obj: override for obj, override in zip(df['obj'], df['override'])}
    return obj2grammar, obj2override

def get_scn2grammar():
    df = pd.read_csv(r'C:\PycharmProjects\SchemeRep\llama\scn_w_grammar.csv')
    scn2grammar = {scn: grammar for scn, grammar in zip(df['scene'], df['grammar'])}
    scn2override = {scn: override for scn, override in zip(df['scene'], df['override'])}
    return scn2grammar, scn2override

def get_llama_activations(obj, scn,
                          activation_model='meta-llama/Llama-3.2-1b',):
    obj2grammar, obj2override = get_obj2grammar()
    scn2grammar, scn2override = get_scn2grammar()

    if not pd.isna(scn2override[scn]):
        scene_part = scn2override[scn]
        scene_part = f'{scene_part},'
    else:
        scene_part = f'{scn2grammar[scn]} {scn},'

    if not pd.isna(obj2override[obj]):
        object_part = obj2override[obj]
    else:
        object_part = f'{obj2grammar[obj]} {obj}'


    sentence = f'{scene_part} {object_part}'
    if scn == 'inside of a car':
        scn = 'car'
    elif scn == 'arch':
        scn = 'desert arch'
    elif scn == 'office space':
        scn = 'office'
    if obj == 'oversize tire':
        obj = 'oversized tire'

    # print(f'{scn=}, {obj=} | {sentence}')

    t = time()
    extractor = get_llama_extractor(model_name=activation_model)
    res = extractor.extract_activations(sentence, [obj, scn], )
    print(f'Time needed for activation extraction: {time() - t:.3f} s')
    print(f'\t{[obj, scn]=} | {sentence=}')
    # quit()
    return res



def get_llama_d_vecs(cat='input', layer_name=1, normalize=True,
                     activation_model='meta-llama/Llama-3.2-1b',
                     all_possible=False, obj_scn_norm=False):
    cat2outer = {'gate_proj_in': 'mlp', 'up_proj_in': 'mlp',
                 'down_proj_in': 'mlp', 'act_fn_in': 'mlp',
                 'gate_proj_out': 'mlp', 'up_proj_out': 'mlp',
                 'down_proj_out': 'mlp', 'act_fn_out': 'mlp',

                 'q_proj': 'attn', 'k_proj': 'attn', 'v_proj': 'attn',
                 'attn_weights': 'attn', 'attn_output': 'attn',
                 'input': 'attn'}
    if cat2outer[cat] == 'mlp':
        if '_in' in cat:
            cat_ = cat.replace('_in', '')
            inner = 'mlp_in'
        else:
            assert '_out' in cat
            cat_ = cat.replace('_out', '')
            inner = 'mlp_out'
    else:
        cat_ = cat
        inner = cat2outer[cat]

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    already_done = set()
    scn_objs = []
    obj_scns = []
    d_vecs = {}
    for sn in sns:
        df_sn = get_trial_info(sn)
        objs = df_sn['obj'].to_list()
        scns = df_sn['scene'].to_list()
        if all_possible:
            objs_ = []
            scns_ = []
            for obj in objs:
                for scn in scns:
                    objs_.append(obj)
                    scns_.append(scn)
            objs = objs_
            scns = scns_

        for obj, scn in zip(objs, scns):
            if (obj, scn) in already_done: continue
            already_done.add((obj, scn))
            scn_objs.append((scn, obj))
            obj_scns.append((obj, scn))
            # print(f'{sn=}, {obj=}, {scn=}')
            # obj = 'oversize tire'
            # scn = 'inside of a car'

            res = pickle_wrap(get_llama_activations,
                              kwargs={'obj': obj, 'scn': scn,
                                      'activation_model': activation_model},
                              easy_override=False,
                              verbose=-1)
            # quit()

            for obj_scn, idx0 in zip(['obj', 'scn'], [0, 1]):
                if cat == 'attn_weights':
                    v = np.nanmean(res['attn']['attn_weights'][layer_name][idx0],
                                   axis=(0, 1))
                else:
                    v = np.nanmean(res[inner][cat_][layer_name][idx0],
                                   axis=0)
                    if len(v.shape) > 1:
                        v = v.reshape(-1)
                if idx0 == 0:
                    d_vecs[(scn, obj)] = v
                else:
                    d_vecs[(obj, scn)] = v

    if obj_scn_norm:
        t_st = time()
        d_vecs = norm_by_obj(d_vecs, cat=cat, layer_name=layer_name,
                             normalize=normalize,
                             activation_model=activation_model)
        print(f'Time needed for normalize by obj & scn: {time() - t_st:.3f} s')
    elif normalize:
        scn_objs_vecs = [d_vecs[(scn, obj)] for scn, obj in scn_objs]
        M = np.nanmean(scn_objs_vecs, axis=0)
        SD = np.nanstd(scn_objs_vecs, axis=0)
        for (scn, obj) in scn_objs:
            d_vecs[(scn, obj)] = (d_vecs[(scn, obj)] - M) / SD
        obj_scns_vecs = [d_vecs[(obj, scn)] for obj, scn in obj_scns]
        M = np.nanmean(obj_scns_vecs, axis=0)
        SD = np.nanstd(obj_scns_vecs, axis=0)
        for (obj, scn) in obj_scns:
            d_vecs[(obj, scn)] = (d_vecs[(obj, scn)] - M) / SD

    return d_vecs

def norm_by_obj(d_vecs, cat='input', layer_name=1, normalize=True,
                activation_model='meta-llama/Llama-3.2-1b',):

    d_vecs_all = get_llama_d_vecs(cat=cat, layer_name=layer_name,
                                  normalize=normalize,
                                  activation_model=activation_model,
                                  all_possible=True)
    d_vecs_obj_l = defaultdict(list)
    for obj, scn in d_vecs_all.keys():
        d_vecs_obj_l[obj].append(d_vecs_all[(obj, scn)])
    d_vecs_obj_M = {obj: np.nanmean(d_vecs_obj_l[obj], axis=0)
                    for obj in d_vecs_obj_l.keys()}
    d_vecs_obj_SD = {obj: np.nanstd(d_vecs_obj_l[obj], axis=0)
                     for obj in d_vecs_obj_l.keys()}
    d_vecs_all_obj_norm = {}
    for (obj, scn), v in d_vecs_all.items():
        d_vecs_all_obj_norm[(obj, scn)] = (v - d_vecs_obj_M[obj]) / d_vecs_obj_SD[obj]

    d_vecs_scn_l = defaultdict(list)
    for obj, scn in d_vecs_all.keys():
        d_vecs_scn_l[scn].append(d_vecs_all[(obj, scn)])
    d_vecs_scn_M = {scn: np.nanmean(d_vecs_scn_l[scn], axis=0)
                    for scn in d_vecs_scn_l.keys()}
    d_vecs_scn_SD = {scn: np.nanstd(d_vecs_scn_l[scn], axis=0)
                     for scn in d_vecs_scn_l.keys()}

    for obj, scn in d_vecs.keys():
        v = d_vecs_all_obj_norm[(obj, scn)]
        v = (v - d_vecs_scn_M[scn]) / d_vecs_scn_SD[scn]
        d_vecs[(obj, scn)] = v
    return d_vecs


@cache
def get_sn_fp_llama_RSM(sn, fp, semantic_tup, dist='spear', within_to_nan=True,
                        normalize=True, #obj_scn_norm=False,
                        ):
    out = pickle_wrap(get_sn_fp_llama_RSM_, kwargs={'sn': sn, 'fp': fp,
                                                    'semantic_tup': semantic_tup,
                                                    'dist': dist, 'within_to_nan': within_to_nan,
                                                    'normalize': normalize,
                                                    # 'obj_scn_norm': obj_scn_norm
                                                    },
                      easy_override=False, verbose=-1)
    return out

# @cache
def get_sn_fp_llama_RSM_(sn, fp, semantic_tup, dist='spear', within_to_nan=True,
                         normalize=True, #obj_scn_norm=False,
                         ):
    activation_model = semantic_tup[4]
    obj_scn_norm = semantic_tup[5]
    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)


    d_vecs = pickle_wrap(get_llama_d_vecs, kwargs={'cat': semantic_tup[1],
                                                   'layer_name': semantic_tup[2],
                                                   'normalize': normalize or obj_scn_norm,
                                                   'activation_model': activation_model,
                                                   'obj_scn_norm': obj_scn_norm},
                         easy_override=False, verbose=-1,
                         RAM_cache=True)

    if semantic_tup[3] == 'obj':
        vecs = [d_vecs[(obj, scn)] for (obj, scn) in zip(df_sn['obj'], df_sn['scene'])]
    else:
        vecs = [d_vecs[(scn, obj)] for (obj, scn) in zip(df_sn['obj'], df_sn['scene'])]
    vecs = np.array(vecs)

    for i in range(vecs.shape[0]):
        num_nan = np.sum(np.isnan(vecs[i]))
        prop_nan = num_nan / vecs.shape[1]
        num_inf = np.sum(np.isinf(vecs[i]))
        prop_inf = num_inf / vecs.shape[1]
        assert prop_nan + prop_inf < .01, \
            f'High non-numbers: {prop_nan=:.2%}, {prop_inf=:.2%}'

        if np.any(np.isinf(vecs[i])):
            vecs[i, np.isinf(vecs[i])] = np.nan
        if np.any(np.isnan(vecs[i])):
            vecs[i, np.isnan(vecs[i])] = np.nanmean(vecs[i])

    if dist == 'corr':
        RSM = np.corrcoef(vecs)
    elif dist == 'spear':
        RSM = stats.spearmanr(vecs, axis=1).correlation
    else:
        raise ValueError

    RSM[np.diag_indices_from(RSM)] = np.nan
    if within_to_nan:
        RSM = within_run_to_nan3(RSM)
    else:
        RSM[np.diag_indices_from(RSM)] = np.nan
    return RSM

def within_run_to_nan3(RDM):
    RDM_ = RDM.copy()
    trial_per_run = RDM.shape[0] // 3
    for run in range(3):
        low = run * trial_per_run
        high = (run + 1) * trial_per_run
        RDM_[low:high, low:high] = np.nan
    # TODO: Fix, this won't work properly except for on encoding!!
    return RDM_

if __name__ == '__main__':

    # norm_by_obj(None)

    semantic_l = []
    all_llama_cats = ['gate_proj_in', 'up_proj_in', 'down_proj_in', 'act_fn_in',
                      'gate_proj_out', 'up_proj_out', 'down_proj_out', 'act_fn_out',
                      'q_proj', 'k_proj', 'v_proj', 'attn_weights', 'attn_output',
                      'input']
    all_llama_layers = list(range(16))

    obj_scn_norm = True
    for llama_cat in all_llama_cats:
        for llama_layer in all_llama_layers:
            semantic_l.append(('llama', llama_cat, llama_layer, 'obj',
                               r'meta-llama/Llama-3.2-3b', obj_scn_norm))
            semantic_l.append(('llama', llama_cat, llama_layer, 'scn',
                               r'meta-llama/Llama-3.2-3b', obj_scn_norm))


    # get_llama_d_vecs(cat='gate_proj_in', layer_name=0)
    for semantic in semantic_l:
        get_sn_fp_llama_RSM(102, 'obj7_fMRI', semantic,
                            dist='spear', within_to_nan=True,
                            normalize=True)
    # quit()

    # get_sn_fp_llama_RSM(102, 'obj7_fMRI',
    #                     ('llama', 'act_fn_in', 14, 'obj'),
    #                     dist='spear', within_to_nan=True,
    #                     normalize=True)


    get_llama_activations('ATM', 'waves')

    # TODO: Need to delete and rerun all except: get_llama_activations
