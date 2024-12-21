
from Utils.pickle_wrap_funcs import pickle_wrap
from llama.BERT_core import BERTLayerActivationExtractor
from llama.get_obj_scn_vecs import get_sentence_obj_scn_in, within_run_to_nan3
from llama.simCSE_core import DetailedSimCSEEmbedder
from organize_bhv import get_trial_info
from functools import cache
import numpy as np
from scipy import stats

RAM_CACHE_BERT = False

@cache
def get_BERT_extractor():
    extractor = BERTLayerActivationExtractor()
    return extractor


def get_BERT_activations(obj, scn):
    extractor = get_BERT_extractor()
    sentence, obj, scn = get_sentence_obj_scn_in(obj, scn)
    activations = extractor.extract_activations(sentence, [obj, scn])
    return activations[0], activations[1]

@cache
def get_simCSE_extractor():
    extractor = DetailedSimCSEEmbedder()
    return extractor

def get_simCSE_activations(obj, scn):
    extractor = get_simCSE_extractor()
    sentence, obj, scn = get_sentence_obj_scn_in(obj, scn)
    activations = extractor.extract_activations(sentence, [obj, scn])
    return activations[0], activations[1]

def get_BERT_d_vecs_non_normed(layer_name=1, code='BERT',
                               all_possible=False):
    sns = ['102', '103', '104'] # everyone else is a duplicate
    d_vecs_obj = {}
    d_vecs_scn = {}
    already_done = set()
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
            if all_possible and len(already_done) % 100 == 0:
                num_done = len(already_done)
                num_total = len(objs)
                p_done = num_done / num_total
                print('-*-*-*-')
                print(f'all_possible progress: {p_done:.1%} ({num_done=}, {num_total=})')

            if code == 'BERT':
                func = get_BERT_activations
            elif code == 'simCSE':
                func = get_simCSE_activations
            else:
                raise ValueError(f'Invalid BERT_d_vecs code: {code=}')
            obj_act, scn_act = pickle_wrap(func,
                                           kwargs={'obj': obj,
                                                   'scn': scn},
                                           RAM_cache=True,)
            obj_vec = obj_act[layer_name]
            scn_vec = scn_act[layer_name]
            d_vecs_obj[(scn, obj)] = obj_vec
            d_vecs_scn[(obj, scn)] = scn_vec
    return d_vecs_obj, d_vecs_scn

def get_BERT_d_vecs(layer_name=1, normalize=True, code='BERT',
                    all_possible=False):
    d_vecs_obj, d_vecs_scn = pickle_wrap(get_BERT_d_vecs_non_normed,
                                         kwargs={'layer_name': layer_name,
                                                 'code': code,
                                                 'all_possible': all_possible},)
    if normalize:
        obj_vecs = np.array([list(d_vecs_obj.values())])[0]
        M = np.nanmean(obj_vecs, axis=0)
        SD = np.nanstd(obj_vecs, axis=0)
        for (scn, obj) in d_vecs_obj.keys():
            d_vecs_obj[(scn, obj)] = (d_vecs_obj[(scn, obj)] - M) / SD

        scn_vecs = np.array([list(d_vecs_scn.values())])[0]
        M = np.nanmean(scn_vecs, axis=0)
        SD = np.nanstd(scn_vecs, axis=0)
        for (obj, scn) in d_vecs_scn.keys():
            d_vecs_scn[(obj, scn)] = (d_vecs_scn[(obj, scn)] - M) / SD
    return d_vecs_obj, d_vecs_scn

def get_sn_fp_BERT_RSM_(sn, fp, scn_obj='obj',
                       layer_name=1, dist='spear', within_to_nan=True,
                       normalize=True, code='BERT'):
    d_vecs_obj, d_vecs_scn = pickle_wrap(get_BERT_d_vecs,
                                         kwargs={'layer_name': layer_name,
                                                 'normalize': normalize,
                                                 'code': code,
                                                 'all_possible': scn_obj == 'obj_M'
                                                 },
                                         verbose=-1)
    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)

    if scn_obj == 'obj':
        ar = np.array([d_vecs_obj[(scn, obj)] for scn, obj in
                       zip(df_sn['scene'], df_sn['obj'])])
    elif scn_obj == 'obj_M':
        scns = df_sn['scene'].to_list()
        ar = []
        for (obj, scn) in zip(df_sn['obj'], df_sn['scene']):
            vecs_obj = []
            for scn2 in scns:
                vecs_obj.append(d_vecs_obj[(scn2, obj)])
            ar.append(np.nanmean(vecs_obj, axis=0))
    elif scn_obj == 'obj_dif':
        scns = df_sn['scene'].to_list()
        vecs = []
        for (obj, scn) in zip(df_sn['obj'], df_sn['scene']):
            vecs_obj = []
            for scn2 in scns:
                vecs_obj.append(d_vecs_obj[(scn2, obj)])
            vec_obj_M = np.nanmean(vecs_obj, axis=0)
            vec_obj = d_vecs_obj[(scn, obj)]
            vecs.append(vec_obj - vec_obj_M)
    else:
        ar = np.array([d_vecs_scn[(obj, scn)] for scn, obj in
                       zip(df_sn['scene'], df_sn['obj'])])

    if dist == 'corr':
        RSM = np.corrcoef(ar)
    elif dist == 'spear':
        RSM = stats.spearmanr(ar, axis=1).correlation
    else:
        raise ValueError

    RSM[np.diag_indices_from(RSM)] = np.nan
    if within_to_nan:
        RSM = within_run_to_nan3(RSM)
    else:
        RSM[np.diag_indices_from(RSM)] = np.nan
    return RSM

def get_sn_fp_BERT_RSM_l(sn, fp, semantic_tup_l, dist='spear', within_to_nan=True,):
    all_RSM = []
    for semantic_tup in semantic_tup_l:
        RSM = get_sn_fp_BERT_RSM(sn, fp, semantic_tup,
                                 dist=dist, within_to_nan=within_to_nan)
        all_RSM.append(RSM)
    out = np.nanmean(all_RSM, axis=0)
    return out

def get_sn_fp_BERT_RSM(sn, fp, semantic, dist='spear', within_to_nan=True,):
    code = semantic[1]
    layer_name = semantic[2]
    scn_obj = semantic[3]
    normalize = semantic[5]
    return pickle_wrap(get_sn_fp_BERT_RSM_,
                       kwargs={'sn': sn, 'fp': fp, 'scn_obj': scn_obj,
                               'layer_name': layer_name, 'dist': dist,
                               'within_to_nan': within_to_nan,
                               'normalize': normalize,
                               'code': code},
                       verbose=-1)

if __name__ == '__main__':
    RAM_CACHE_BERT = True
    get_BERT_d_vecs_non_normed()