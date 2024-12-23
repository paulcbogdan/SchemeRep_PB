import pandas as pd

from functools import cache
from time import time

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.old.gen_sentence import generate_sentences_API
from organize_bhv import get_trial_info
import numpy as np
from scipy import stats
from collections import defaultdict
import pickle

RAM_CACHE_LLAMA = False

@cache
def get_llama_extractor(model_name='meta-llama/Llama-3.2-1b'):
    from llama.llama_test import LlamaActivationExtractor
    extractor = LlamaActivationExtractor(model_name)
    extractor._register_comprehensive_hooks()
    return extractor

@cache
def get_obj2grammar():
    df = pd.read_csv(r'C:\PycharmProjects\SchemeRep\llama\obj_w_grammar.csv')
    obj2grammar = {obj: grammar for obj, grammar in zip(df['obj'], df['grammar'])}
    obj2override = {obj: override for obj, override in zip(df['obj'], df['override'])}
    return obj2grammar, obj2override

@cache
def get_scn2grammar():
    df = pd.read_csv(r'C:\PycharmProjects\SchemeRep\llama\scn_w_grammar.csv')
    scn2grammar = {scn: grammar for scn, grammar in zip(df['scene'], df['grammar'])}
    scn2override = {scn: override for scn, override in zip(df['scene'], df['override'])}
    return scn2grammar, scn2override

def get_sentence_obj_scn_in(obj, scn):
    obj2grammar, obj2override = get_obj2grammar()
    scn2grammar, scn2override = get_scn2grammar()
    schemerep_objs = get_schemerep_objs()
    deve = obj not in schemerep_objs
    if deve:
        if obj.lower() in ['a', 'e', 'i', 'o', 'u']:
            object_part = f'an {obj}'
        else:
            object_part = f'a {obj}'
    elif not pd.isna(obj2override[obj]):
        object_part = obj2override[obj]
    else:
        object_part = f'{obj2grammar[obj]} {obj}'

    if scn is None:
        if obj == 'oversize tire':
            obj = 'oversized tire'
        if object_part[:3] == 'an ':
            object_part = 'An ' + object_part[3:]
        elif object_part[:2] == 'a ':
            object_part = 'A ' + object_part[2:]
        return object_part, obj, None

    if not pd.isna(scn2override[scn]):
        scene_part = scn2override[scn]
        scene_part = f'{scene_part},'
    else:
        scene_part = f'{scn2grammar[scn]} {scn},'

    sentence = f'{scene_part} {object_part}'
    if scn == 'inside of a car':
        scn = 'car'
    elif scn == 'arch':
        scn = 'desert arch'
    elif scn == 'office space':
        scn = 'office'
    if obj == 'oversize tire':
        obj = 'oversized tire'

    return sentence, obj, scn

@cache
def get_schemerep_objs():
    df_sn = get_trial_info('102')
    objs = df_sn['obj'].unique()
    return set(objs)

def get_llama_activations(obj, scn,
                          activation_model='meta-llama/Llama-3.2-1b',):

    if isinstance(activation_model, tuple):
        if activation_model[1] == 'grok_first':
            activation_model = activation_model[0]
            sentences = pickle_wrap(generate_sentences_API,
                                    kwargs={'words': [scn, obj]},
                                    )
            # sentences_ = []
            reses = []
            print(f'Sentences set: [{scn}, {obj}]')
            for sentence in sentences:
                print(f'DO: {sentence=}')
                if 'bartender' in sentence:
                    idx_scn1 = sentence.lower().index(f'{scn} '.lower())
                else:
                    idx_scn1 = sentence.lower().index(scn.lower())
                if obj == 'dragonfly' and 'dragonflies' in sentence.lower():
                    obj = 'dragonflies'
                if obj == 'dragonflies' and 'dragonfly' in sentence.lower():
                    obj = 'dragonfly'
                if obj == 'cactus' and 'cacti' in sentence.lower():
                    obj = 'cacti'
                if obj == 'cacti' and 'cactus' in sentence.lower():
                    obj = 'cactus'
                # if obj not in sentence: print(f'MISSING OBJ: {obj=}, {sentence=}')
                idx_obj0 = sentence.lower().index(obj.lower())
                if (obj == 'football') and (scn == 'football field'):
                    sentence_chopped = sentence
                elif idx_obj0 < idx_scn1:
                    sentence_chopped = sentence[:idx_scn1 + len(scn)]
                else:
                    sentence_chopped = sentence[:idx_obj0 + len(obj)]
                t_st = time()
                extractor = get_llama_extractor(model_name=activation_model)
                res_ = pickle_wrap(extractor.extract_activations,
                                   kwargs={'sentence': sentence_chopped,
                                           'target_words': [obj, scn]},
                                   easy_override=False, verbose=-1,
                                   )
                # res_ = extractor.extract_activations(sentence_chopped, [obj, scn], )
                print(f'\tTime needed for activation extraction: {time() - t_st:.3f} s |'
                      f'{[obj, scn]} | {sentence=}')
                reses.append(res_)

            res = {}
            for outer in ['attn', 'mlp_out', 'mlp_in']: # this averages across the 8 sentences
                res[outer] = {}
                for inner in reses[0][outer].keys():
                    res[outer][inner] = {}
                    for layer_num in reses[0][outer][inner].keys():
                        res[outer][inner][layer_num] = []
                        for idx in range(2):
                            reses_val = []
                            for res_ in reses:
                                if inner == 'attn_weights':
                                    reses_val.append(np.nanmean(res_[outer][inner][layer_num][idx],
                                                                axis=(0, 1), keepdims=True))
                                else:
                                    reses_val.append(np.nanmean(res_[outer][inner][layer_num][idx],
                                                                axis=0, keepdims=True))
                            res[outer][inner][layer_num].append(np.nanmean(reses_val, axis=0))
        else:
            raise ValueError
    else:
        sentence, obj, scn = get_sentence_obj_scn_in(obj, scn)
        t = time()
        extractor = get_llama_extractor(model_name=activation_model)
        if scn is None:
            print(f'Object solo sentence: {sentence=}')
            res = extractor.extract_activations(sentence, [obj], )
        else:
            res = extractor.extract_activations(sentence, [obj, scn], )
        print(f'Time needed for activation extraction: {time() - t:.3f} s')
        print(f'\t{[obj, scn]=} | {sentence=}')
    return res


@cache
def process_cat_cat_inner(cat):
    if cat in ['BERT', 'simCSE']:
        return cat, cat
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
    return cat_, inner

def get_llama_d_vecs_non_normed(cat='input', layer_name=1,
                                activation_model='meta-llama/Llama-3.2-1b',
                                all_possible=False):
    last_only = False
    if isinstance(layer_name, tuple):
        last_only = layer_name[1]
        layer_name = layer_name[0]
    if isinstance(activation_model, tuple) and activation_model[1] != 'grok_first':
        assert activation_model[1] == 'obj_solo'
        obj_solo = True
        activation_model = activation_model[0]
        assert not all_possible
    else:
        obj_solo = False
    cat_, inner = process_cat_cat_inner(cat)
    print('getting llama d_vecs no norming...')

    sns = ['102', '103', '104'] # everyone else is a duplicate
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
            if obj_solo: scn = None

            if (obj, scn) in already_done: continue
            already_done.add((obj, scn))
            scn_objs.append((scn, obj))
            obj_scns.append((obj, scn))
            if all_possible and len(already_done) % 100 == 0:
                num_done = len(already_done)
                num_total = len(objs)
                p_done = num_done / num_total
                print('-*-*-*-')
                print(f'all_possible progress: {p_done:.1%} ({num_done=}, {num_total=})')
                # print('-*-*-*-')

            t_st = time()

            res, fp_pkl = pickle_wrap(get_llama_activations,
                              kwargs={'obj': obj, 'scn': scn,
                                      'activation_model': activation_model},
                              easy_override=False, verbose=-1, dir_branches=100,
                              RAM_cache=RAM_CACHE_LLAMA, get_fp=True
                              )
            if 'mlp_out' in res and 'down_proj' in res['mlp_out']:
                del res['mlp_out']['down_proj']
                del res['mlp_out']['up_proj']
                del res['mlp_out']['gate_proj']
                del res['mlp_in']['down_proj']
                del res['mlp_in']['up_proj']
                del res['attn']['q_proj']
                del res['attn']['k_proj']
                with open(fp_pkl, 'wb') as f:
                    pickle.dump(res, f)

            for idx_target in [0, 1]:
                if obj_solo and idx_target == 1: continue
                # ... = extractor.extract_activations(sentence, [obj, scn], )
                # idx0 is the object passed
                # idx1 is the scene passed
                if cat == 'attn_weights':
                    if last_only:
                        v = res['attn']['attn_weights'][layer_name][idx_target][-1, -1]
                    else:
                        v = np.nanmean(res['attn']['attn_weights'][layer_name][idx_target],
                                       axis=(0, 1))
                else:
                    if last_only:
                        v = res[inner][cat_][layer_name][idx_target][-1]
                    else:
                        v = np.nanmean(res[inner][cat_][layer_name][idx_target], axis=0)
                    if len(v.shape) > 1:
                        v = v.reshape(-1) # reshapes q_proj, k_proj, v_proj
                                          # which are (attn_heads, vector)
                if idx_target == 0:
                    d_vecs[(scn, obj)] = v
                else:
                    d_vecs[(obj, scn)] = v
                    # SECOND ENTRY IS THE TARGET ONE
        if all_possible:
            break


    return d_vecs, scn_objs, obj_scns



def get_llama_d_vecs(cat='input', layer_name=1, normalize=True,
                     activation_model='meta-llama/Llama-3.2-1b',
                     all_possible=False):
    d_vecs, scn_objs, obj_scns = (
        pickle_wrap(get_llama_d_vecs_non_normed,
                    kwargs={'cat': cat, 'layer_name': layer_name,
                            'activation_model': activation_model,
                            'all_possible': all_possible},
                    easy_override=False, verbose=-1, RAM_cache=True))

    if not isinstance(normalize, bool):
        t_st = time()
        d_vecs = norm_by_obj(d_vecs, cat=cat, layer_name=layer_name,
                             norm_axis=normalize,
                             activation_model=activation_model)
        print(f'Time needed for normalize by obj and/or scn ({normalize}): '
              f'{time() - t_st:.3f} s')
    elif normalize:
        scn_objs_vecs = [d_vecs[(scn, obj)] for scn, obj in scn_objs]
        M = np.nanmean(scn_objs_vecs, axis=0)
        SD = np.nanstd(scn_objs_vecs, axis=0)
        for (scn, obj) in scn_objs:
            d_vecs[(scn, obj)] = (d_vecs[(scn, obj)] - M) / SD
        if isinstance(activation_model, tuple) and activation_model[1] == 'obj_solo':
            pass
        else:
            obj_scns_vecs = [d_vecs[(obj, scn)] for obj, scn in obj_scns]
            M = np.nanmean(obj_scns_vecs, axis=0)
            SD = np.nanstd(obj_scns_vecs, axis=0)
            for (obj, scn) in obj_scns:
                d_vecs[(obj, scn)] = (d_vecs[(obj, scn)] - M) / SD

    return d_vecs

def norm_by_obj(d_vecs, cat='input', layer_name=1, normalize=True,
                activation_model='meta-llama/Llama-3.2-1b',
                norm_axis=(0, 1)):

    # NORM AXIS = 0 means you are standardizing with respect to all other targets toward the context
    #    so if the context is "beach" and the target (idx0 = 0) is "ball", you are taking "ball"
    #    and substracting the mean for all targets at beach then dividing by the standard deviation
    # NORM AXIS = 1 means you are standardizing with respect to all other contexts toward the target
    # NORM AXIS = (0, 1) means you are doing both

    d_vecs_all = pickle_wrap(get_llama_d_vecs, kwargs={'cat': cat,
                                                       'layer_name': layer_name,
                                                       'normalize': False,
                                                       'activation_model': activation_model,
                                                       'all_possible': True},
                                easy_override=False, verbose=-1,
                                RAM_cache=False)

    assert np.all(np.abs(d_vecs[('surfing board', 'waves')] -
                         d_vecs_all[('surfing board', 'waves')])) < 1e-6
    if norm_axis == 0:
        d_vecs_obj_l = defaultdict(list)
        for obj, scn in d_vecs_all.keys():
            d_vecs_obj_l[obj].append(d_vecs_all[(obj, scn)])
            # Divide the second one by all others of the second one of the same first one
        d_vecs_obj_M = {obj: np.nanmean(d_vecs_obj_l[obj], axis=0)
                        for obj in d_vecs_obj_l.keys()}
        # print(d_vecs_obj_M.shape)
        # quit()
        d_vecs_obj_SD = {obj: np.nanstd(d_vecs_obj_l[obj], axis=0)
                         for obj in d_vecs_obj_l.keys()}
        for (obj, scn), v in d_vecs.items():
            d_vecs[(obj, scn)] = (v - d_vecs_obj_M[obj]) / d_vecs_obj_SD[obj]
        return d_vecs
    elif norm_axis == 1:
        d_vecs_obj_l = defaultdict(list)
        for obj, scn in d_vecs_all.keys():
            d_vecs_obj_l[scn].append(d_vecs_all[(obj, scn)])
        d_vecs_obj_M = {scn: np.nanmean(d_vecs_obj_l[scn], axis=0)
                        for scn in d_vecs_obj_l.keys()}
        d_vecs_obj_SD = {scn: np.nanstd(d_vecs_obj_l[scn], axis=0)
                         for scn in d_vecs_obj_l.keys()}
        for (obj, scn), v in d_vecs.items():
            d_vecs[(obj, scn)] = (v - d_vecs_obj_M[scn]) / d_vecs_obj_SD[scn]
        return d_vecs
    else:
        d_vecs_obj_l = defaultdict(list)
        for obj, scn in d_vecs_all.keys():
            d_vecs_obj_l[obj].append(d_vecs_all[(obj, scn)])
            # Divide the second one by all others of the second one of the same first one
        d_vecs_obj_M = {obj: np.nanmean(d_vecs_obj_l[obj], axis=0)
                        for obj in d_vecs_obj_l.keys()}
        d_vecs_obj_SD = {obj: np.nanstd(d_vecs_obj_l[obj], axis=0)
                         for obj in d_vecs_obj_l.keys()}

        assert norm_axis == (0, 1)
        d_vecs_all_obj_norm = {}
        for (obj, scn), v in d_vecs_all.items():
            d_vecs_all_obj_norm[(obj, scn)] = (v - d_vecs_obj_M[obj]) / d_vecs_obj_SD[obj]

        d_vecs_scn_l = defaultdict(list)
        for obj, scn in d_vecs_all_obj_norm.keys():
            d_vecs_scn_l[scn].append(d_vecs_all_obj_norm[(obj, scn)])


        d_vecs_scn_M = {scn: np.nanmean(d_vecs_scn_l[scn], axis=0)
                        for scn in d_vecs_scn_l.keys()}
        d_vecs_scn_SD = {scn: np.nanstd(d_vecs_scn_l[scn], axis=0)
                         for scn in d_vecs_scn_l.keys()}

        for (key0, key1) in d_vecs.keys():
            v = d_vecs_all_obj_norm[(key0, key1)]
            v = (v - d_vecs_scn_M[key1]) / d_vecs_scn_SD[key1]
            d_vecs[(key0, key1)] = v
        return d_vecs

def get_sn_fp_llama_RSM_l(sn, fp, semantic_tup_l, dist='spear', within_to_nan=True,
                          ):
    all_RSM = []
    do_PCA = False
    for semantic_tup in semantic_tup_l:
        if semantic_tup[1] == 'PCA':
            do_PCA = True
            continue

        RSM = get_sn_fp_llama_RSM(sn, fp, semantic_tup, dist=dist, within_to_nan=within_to_nan)
        all_RSM.append(RSM)

    if do_PCA:
        all_RSMS_flat = []
        trils = np.tril_indices(114, k=-1)
        for RSM in all_RSM:
            all_RSMS_flat.append(RSM[trils])
        all_RSMS_flat = np.array(all_RSMS_flat)
        nan_cols = np.any(np.isnan(all_RSMS_flat), axis=0)
        all_RSMS_flat = all_RSMS_flat[:, ~nan_cols]
        from sklearn.decomposition import PCA
        all_RSMS_flat = stats.zscore(all_RSMS_flat, axis=1, nan_policy='omit')


        pca = PCA()
        # pca_result = pca.fit(all_RSMS_flat.T)
        out = pca.fit_transform(all_RSMS_flat.T)
        trils = np.array(trils)
        trils = trils[:, ~nan_cols]
        PCA_RSMs = []
        for i in range(semantic_tup_l[0][2]):
            PCA_RSM = np.full((114, 114), np.nan)
            PCA_RSM[trils[0], trils[1]] = out[:, i]
            PCA_RSMs.append(PCA_RSM)
        all_RSM = PCA_RSMs
        print(f'{np.array(PCA_RSMs).shape=}')

    out = np.nanmean(all_RSM, axis=0)
    return out


@cache
def get_sn_fp_llama_RSM(sn, fp, semantic_tup, dist='spear', within_to_nan=True,
                        ):
    out = pickle_wrap(get_sn_fp_llama_RSM_, kwargs={'sn': sn, 'fp': fp,
                                                    'semantic_tup': semantic_tup,
                                                    'dist': dist,
                                                    'within_to_nan': within_to_nan,
                                                    },
                      easy_override=False, verbose=-1)
    return out

# @cache
def get_sn_fp_llama_RSM_(sn, fp, semantic_tup, dist='spear', within_to_nan=True,
                         # normalize=True, #obj_scn_norm=False,
                         ):
    activation_model = semantic_tup[4]
    if semantic_tup[3] == 'obj_solo':
        activation_model = (activation_model, 'obj_solo')
    normalize = semantic_tup[5]
    # obj_scn_norm = semantic_tup[5]
    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

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
    if semantic_tup[3] == 'obj_solo':
        vecs = []
        for obj in df_sn['obj']:
            vecs.append(d_vecs[(None, obj)])
    elif semantic_tup[3] == 'obj_M':
        scns = df_sn['scene'].to_list()
        vecs = []
        for (obj, scn) in zip(df_sn['obj'], df_sn['scene']):
            vecs_obj = []
            for scn2 in scns:
                vecs_obj.append(d_vecs[(scn2, obj)])
            vecs.append(np.nanmean(vecs_obj, axis=0))
    elif semantic_tup[3] == 'scn_M':
        objs = df_sn['obj'].to_list()
        vecs = []
        for (obj, scn) in zip(df_sn['obj'], df_sn['scene']):
            vecs_scn = []
            for obj2 in objs:
                vecs_scn.append(d_vecs[(obj2, scn)])
            vecs.append(np.nanmean(vecs_scn, axis=0))
    elif semantic_tup[3] == 'obj_dif':
        scns = df_sn['scene'].to_list()
        vecs = []
        for (obj, scn) in zip(df_sn['obj'], df_sn['scene']):
            vecs_obj = []
            for scn2 in scns:
                vecs_obj.append(d_vecs[(scn2, obj)])
            assert len(vecs_obj) > 2
            vec_obj_M = np.nanmean(vecs_obj, axis=0)
            vec_obj = d_vecs[(scn, obj)]
            vecs.append(vec_obj - vec_obj_M)
            num_nans_obj = np.sum(np.isnan(vec_obj))
            num_nans_obj_M = np.sum(np.isnan(vec_obj_M))
            if num_nans_obj > 10 or num_nans_obj_M > 10: # producing reduce warning
                print(f'When preparing obj_dif ({scn}, {obj}):')
                print(f'\tnum_nans_obj: {num_nans_obj}')
                print(f'\tnum_nans_obj_M: {num_nans_obj_M}')
                # raise ValueError
    elif semantic_tup[3] == 'prod':
        vecs = [d_vecs[(obj, scn)] * d_vecs[(scn, obj)]
                for (obj, scn) in zip(df_sn['obj'], df_sn['scene'])]
    elif semantic_tup[3] == 'sum':
        vecs = [d_vecs[(obj, scn)] + d_vecs[(scn, obj)]
                for (obj, scn) in zip(df_sn['obj'], df_sn['scene'])]
    elif semantic_tup[3] == 'obj':
        # for non-attn_weights, to get the object representation, you specify (_, obj)
        # for non-attn_weights, to get the scene representation, you specify (scn, _)
        # for attn_weights, to get the effect of the object on the scene, you specify (scn, obj)
        # for attn_weights, to get the effect of the scene on the object, you specify (obj, scn)

        # TODO: When I redo everything flip this
        vecs = [d_vecs[(scn, obj)] for (obj, scn) in zip(df_sn['obj'], df_sn['scene'])]
        # vecs = [d_vecs[(obj, scn)] for (obj, scn) in zip(df_sn['obj'], df_sn['scene'])]
    else:
        vecs = [d_vecs[(obj, scn)] for (obj, scn) in zip(df_sn['obj'], df_sn['scene'])]
        # vecs = [d_vecs[(scn, obj)] for (obj, scn) in zip(df_sn['obj'], df_sn['scene'])]
    vecs = np.array(vecs)
    # print(vecs)
    # quit()

    # thresholds_met = {0.1: False, 0.2: False, 0.3: False, 0.4: False, 0.5: False}
    M_nans = []
    M_infs = []
    for i in range(vecs.shape[0]):
        num_nan = np.sum(np.isnan(vecs[i]))
        prop_nan = num_nan / vecs.shape[1]
        num_inf = np.sum(np.isinf(vecs[i]))
        prop_inf = num_inf / vecs.shape[1]
        # if prop_nan + prop_inf > .6:
        #     f'High non-numbers: {prop_nan=:.2%}, {prop_inf=:.2%}'
        M_nans.append(prop_nan)
        M_infs.append(prop_inf)

        if np.any(np.isinf(vecs[i])):
            vecs[i, np.isinf(vecs[i])] = np.nan
        if np.any(np.isnan(vecs[i])):
            vecs[i, np.isnan(vecs[i])] = np.nanmean(vecs[i])

    M_nan_overall = np.mean(M_nans)
    M_inf_overall = np.mean(M_infs)
    if sn == 102 and fp == 'obj7_fMRI':
        print(f'Overall NaN: {M_nan_overall:.2%}, Inf: {M_inf_overall=:.2%} | {semantic_tup}')

    # if isinstance(semantic_tup[1], tuple):
    #     vecs_ = []
    # else:
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
    SEMANTIC_L = []

    # redundant: act_fn_in, up_proj_in
    all_llama_cats = ['gate_proj_in', 'down_proj_in',
                      'gate_proj_out', 'up_proj_out', 'down_proj_out', 'act_fn_out',
                      'q_proj', 'k_proj', 'v_proj', 'attn_weights', 'attn_output',
                      'input']

    # all_llama_layers = list(range(28))
    # all_llama_layers = list(range(16, 28))

    # NORMALIZE = False
    # MODEL = r'meta-llama/Llama-3.1-3b' # 16 layers, 2k vectors
    MODEL = r'meta-llama/Llama-3.2-3b' # 28?? layers, 4k vectors?? (double check numbers)
    # MODEL = r'meta-llama/Llama-3.1-70b' # 80 layers, 8k vectors
    # MODEL = r'meta-llama/Llama-3.3-70b-Instruct' # 80 layers, 8k vectors
    # MODEL = r'meta-llama/Llama-2-7b-hf' # 32 layers, 32x128 vectors
    all_llama_layers = list(range(0, 80 if '70b' in MODEL else 28))

    all_llama_cats = ['gate_proj_in']
    # all_llama_cats = ['attn_weights']
    # all_llama_cats = ['attn_output']

    # all_llama_cats = ['gate_proj_in', 'attn_weights']

    # MODEL = (MODEL, 'grok_first')
    NORMALIZE = True
    LAST_ONLY = False

    for LLAMA_CAT in all_llama_cats:
        for LLAMA_LAYER in all_llama_layers:
            # if LLAMA_CAT in ['attn_weights', 'attn_output']:
                # SEMANTIC_L.append(('llama', LLAMA_CAT,
                #                    (LLAMA_LAYER, True) if LAST_ONLY else LLAMA_LAYER,
                #                    'scn', MODEL, NORMALIZE))
            SEMANTIC_L.append(('llama', LLAMA_CAT,
                               (LLAMA_LAYER, True) if LAST_ONLY else LLAMA_LAYER,
                               'obj_solo', MODEL, NORMALIZE))

    # for LLAMA_CAT in all_llama_cats:
    #     for LLAMA_LAYER in all_llama_layers:
    #         SEMANTIC_L.append(('llama', LLAMA_CAT, (LLAMA_LAYER, True), 'obj_M', MODEL,
    #                            NORMALIZE))
    #         SEMANTIC_L.append(('llama', LLAMA_CAT, (LLAMA_LAYER, True), 'scn_M', MODEL,
    #                            NORMALIZE))

    RAM_CACHE_LLAMA = True

    for SEMANTIC in SEMANTIC_L:
        t_st_setting = time()
        get_sn_fp_llama_RSM(104, 'obj7_fMRI', SEMANTIC,
                            dist='spear', within_to_nan=True,
                            )
        print(f'Time needed to execute setting: {time() - t_st_setting:.3f} s')
