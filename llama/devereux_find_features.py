import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from numba import njit

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_analysis import get_devereux_RSM_by_type
from llama.devereux_llama import get_deve_llama_RSM, get_standard_items_list, get_llama_d_vecs_deve, get_llama_vecs_ar
from llama.model_settings import get_explore_llama
from time import time

# TODO: Find individual features that benefit from contextualization with RSA
# Use get_deve_llama_RSM_(quick=tuple) to get RSM based on difference
# TODO: Then, find items that are driving this RSA effect, max(IRAF)
# Just do the IRAFs, shouldn't be hard
# TODO: Then, find what context produces the max IRAF?
# For each context, measure its distance to the others' vectors at feature = 1 vs. feature = 0

# TODO BEFORE:
#  look at attn_output
#  look at position=1
def get_item_features(item, pf_thresh=300):
    items, df = get_standard_items_list(pf_thresh=pf_thresh,
                                        norm_per_concept=False)
    df_item = df[df['concept'] == item]
    return df_item['feature'].unique()

def get_feature_binary_RSM(pf_thresh=300, feature_type='functional',
                           req_pf=2, req_pf_overall=100,
                           req_items=25):
    items, df = get_standard_items_list(pf_thresh=pf_thresh,
                                        norm_per_concept=False)
    df = df[df['feature type'] == feature_type]
    df = df[df['pf'] >= req_pf]

    pf_feature_cnt = df.groupby('feature')['pf'].sum()
    features = pf_feature_cnt[pf_feature_cnt >= req_pf_overall].index
    items_all = set(df['concept'].unique())
    feat2RSMs = {}
    feature2vecs = {}
    for feature in features:
        items_w_feature = set(df[df['feature'] == feature]['concept'].unique())
        if len(items_w_feature) < req_items:
            continue
        items_wo_feature = set(items_all) - items_w_feature
        # print(f'- {feature} -')
        # print(f'Number of items with feature ({feature} = 1): {len(items_w_feature)}')
        # print(f'Number of items without feature ({feature} = 0): {len(items_wo_feature)}')

        RSM_feat = np.empty((len(items), len(items)))
        for i, item_i in enumerate(items):
            for j, item_j in enumerate(items):
                if i == j:
                    RSM_feat[i, i] = np.nan
                elif i > j:
                    continue
                if item_i in items_w_feature and item_j in items_w_feature:
                    RSM_feat[i, j] = 1
                    RSM_feat[j, i] = 1
                elif item_i in items_wo_feature and item_j in items_wo_feature:
                    RSM_feat[i, j] = 1
                    RSM_feat[j, i] = 1
                else:
                    RSM_feat[i, j] = 0
                    RSM_feat[j, i] = 0
        feat2RSMs[feature] = RSM_feat
        vec = []
        for item in items:
            if item in items_w_feature:
                vec.append(1)
            else:
                vec.append(0)
        feature2vecs[feature] = vec
    return items, feature2vecs, feat2RSMs


def get_llama_IRAFs(RSM, RSM_feat):
    fMRI_RDM_r = stats.rankdata(RSM, axis=0, nan_policy='omit')
    stim_RDM_r = stats.rankdata(RSM_feat, axis=0, nan_policy='omit')
    fMRI_RDM_r = stats.zscore(fMRI_RDM_r, axis=0, nan_policy='omit')
    stim_RDM_r = stats.zscore(stim_RDM_r, axis=0, nan_policy='omit')
    IRAFs = np.nanmean(fMRI_RDM_r * stim_RDM_r, axis=0)
    IRAFs = np.arctanh(IRAFs)
    return IRAFs


def features_benefiting_from_context(st=4, end=14, pf_thresh=300,
                                     activation_model='meta-llama/Llama-3.2-3b',
                                     feature_type='functional'):
    items, feature2vecs, feature2RSMs = (
        get_feature_binary_RSM(pf_thresh=pf_thresh, feature_type=feature_type))

    models = get_explore_llama(activation_model=activation_model,
                               attn=False, st=6, end=7)
    RSM = get_deve_llama_RSM(models, pf_thresh=pf_thresh,
                             quick=(None, 1), item_standard='deve',
                             position=None,
                             )

    # RSM = get_deve_llama_RSM(models, pf_thresh=pf_thresh,
    #                          quick=None, #item_standard='deve',
    #                          position=None,
    #                          )

    trils = np.tril_indices_from(RSM, k=-1)
    RSM_flat = RSM[trils]

    type2RSM = get_devereux_RSM_by_type(pf_thresh=pf_thresh,
                                        attn=False)
    feature2RSMs = type2RSM

    for feature, RSM_feat in feature2RSMs.items():
        # if feature not in ['functional',
        #                    ]: continue

        RSM_feat_flat = RSM_feat[trils]
        RSM_feat_nans = np.isnan(RSM_feat_flat)
        num_nans = np.sum(RSM_feat_nans)
        RSM_flat_ = RSM_flat[~RSM_feat_nans]
        RSM_feat_flat_ = RSM_feat_flat[~RSM_feat_nans]
        r, p = stats.spearmanr(RSM_flat_, RSM_feat_flat_,
                               nan_policy='omit' if num_nans > 0 else 'raise')

        prop_nan = num_nans / len(RSM_feat_flat)
        print(f'- {feature=} -')
        if prop_nan > 0:
            print(f'\t{feature_type=}: {r=:.3f}, {p=:.3f} ({prop_nan=:.1%})')
        else:
            print(f'\t{feature_type=}: {r=:.3f}, {p=:.3f}')

        IRAFs = get_llama_IRAFs(RSM, RSM_feat)
        IRAFs *= 1000
        IRAF_cutoff = np.nanquantile(IRAFs, 0.95)
        # for item, feat_val, IRAF in zip(items, feature2vecs[feature], IRAFs):
        #     if IRAF < IRAF_cutoff:
        #         continue
        #     print(f'{item} ({feat_val}) {IRAF=:.1f}')

        for item, IRAF in zip(items, IRAFs):
            if IRAF < IRAF_cutoff:
                continue
            print(f'{item}: {IRAF=:.1f}')

        # IRAF is reliably higher among feat_val=1

    # harp: IRAF=462.9
    # scissors: IRAF=441.5

def get_vecs_alt_context(items, d_vecs, vecs_all_M, zscore=True,
                         position=None):
    vecs_alt = []
    for context in items:
        if context == 'cigar':
            vecs_alt_context = np.full_like(vecs_alt[-1], np.nan)
            vecs_alt.append(vecs_alt_context)
            continue
        vecs_alt_context = []
        for i, item1 in enumerate(items):
            if item1 == context:
                vec1 = np.full(3072, np.nan)
            elif item1 == 'cigar':
                vec1 = np.full(3072, np.nan)
            else:
                vec1 = d_vecs[(context, item1, item1)]
                if position is None:
                    vec1_flip = d_vecs[(item1, context, item1)]
                    vec1 = np.nanmean([vec1, vec1_flip], axis=0)
            vec1 -= vecs_all_M[i]
            vecs_alt_context.append(vec1)
        vecs_alt.append(vecs_alt_context)
    vecs_alt = np.array(vecs_alt)
    if zscore:
        for i in range(vecs_alt.shape[0]):
            vecs_alt[i] = stats.rankdata(vecs_alt[i], axis=0, nan_policy='omit')
            vecs_alt[i] = stats.zscore(vecs_alt[i], axis=0, nan_policy='omit')
    return vecs_alt


def find_context_helper(target_item='lobster', pf_thresh=300,
                        cat='gate_proj_in', layer_name=9,
                        activation_model='meta-llama/Llama-3.2-3b',
                        normalize=True, quick=None,
                        feature_type='functional',
                        context_specific=True,
                        position=1, symmetric=False):
    # vecs_all_M = get_llama_vecs_ar(pf_thresh, cat, layer_name,
    #                                activation_model, normalize,
    #                                quick=None)
    items, df = get_standard_items_list(pf_thresh=pf_thresh,
                                        norm_per_concept=False)
    vecs_all_M = pickle_wrap(get_llama_vecs_ar,
                             kwargs={'pf_thresh': pf_thresh, 'cat': cat,
                                     'layer_name': layer_name,
                                     'activation_model': activation_model,
                                     'normalize': normalize,
                                     'quick': quick,
                                     'symmetric': symmetric
                                     },
                             easy_override=False, verbose=-1)
    try:
        idx_target = list(items).index(target_item)
    except ValueError:
        return [np.nan]

    d_vecs = pickle_wrap(get_llama_d_vecs_deve,
                         kwargs={'pf_thresh': pf_thresh, 'cat': cat,
                                 'layer_name': layer_name,
                                 'activation_model': activation_model,
                                 'normalize': normalize,
                                 'quick': quick,
                                 'symmetric': symmetric
                                 },
                         easy_override=False, verbose=-1)

    # vecs_alt = []

    vec_len = len(d_vecs[(items[0], target_item, target_item)])

    vecs_target_c = []
    for context in items:
        try:
            if context == target_item:
                vec = np.full(vec_len, np.nan)
            else:
                vec = d_vecs[(context, target_item, target_item)]
                if position is None:
                    vec_flip = d_vecs[(target_item, context, target_item)]
                    vec = np.nanmean([vec, vec_flip], axis=0)
                vec -= vecs_all_M[idx_target]
        except KeyError:
            vec = np.full(vec_len, np.nan)
            print(f'Bad no target match: {context}')
        vecs_target_c.append(vec)
    vecs_target_c = np.array(vecs_target_c)


    # t_st = time()
    if context_specific:
        print('Getting vecs alt')
        fp = 'cache/vecs_alt_context.pkl'
        vecs_all_M = pickle_wrap(get_vecs_alt_context, fp,
                                 kwargs={'items': items, 'd_vecs': d_vecs,
                                         'vecs_all_M': vecs_all_M,
                                         'zscore': True, 'position': position},
                                 easy_override=False, verbose=-1)
        print('Got vecs alt z-scored')
    vecs_target_c = stats.rankdata(vecs_target_c, axis=0, nan_policy='omit')
    vecs_target_c = stats.zscore(vecs_target_c, axis=0, nan_policy='omit')
    if len(vecs_all_M.shape) == 3:
        # print('Z-scored')
        corr = numba_corr_by_context(vecs_target_c, vecs_all_M, )#, nan_mask)
    else:
        vecs_all_M = stats.rankdata(vecs_all_M, axis=0, nan_policy='omit')
        vecs_all_M = stats.zscore(vecs_all_M, axis=0, nan_policy='omit')
        # corr = numba_corr(vecs_all_M, vecs_target_c)#, nan_mask)
        corr = numba_corr(vecs_target_c, vecs_all_M, )#, nan_mask)

    # nan_mask = np.isnan(vecs_all_M) | np.isnan(vecs_target_c)
    corr[np.diag_indices_from(corr)] = np.nan
    # plt.imshow(corr)
    # plt.colorbar()
    # plt.show()

    type2RSM = get_devereux_RSM_by_type(pf_thresh=pf_thresh,
                                        attn=False, )

    items, feature2vecs, feature2RSMs = (
        get_feature_binary_RSM(pf_thresh=pf_thresh,
                               feature_type=feature_type,
                               req_pf=1,
                               req_pf_overall=100,
                               req_items=20
                               ))
    features = get_item_features(target_item, pf_thresh=pf_thresh,
                                 )
    features = [feature for feature in features if feature in feature2vecs]


    # features = ['functional', ]
    # features = ['visual perceptual', 'encyclopedic',
    #             'functional', 'taxonomic',
    #             'other perceptual']

    rs = []
    for feature in features:
        if feature in feature2RSMs:
            RSM_func = feature2RSMs[feature]
        else:
            RSM_func = type2RSM[feature]

        IRAFs = get_llama_IRAFs(corr, RSM_func)
        IRAFs *= 1000
        IRAF_cutoff = np.nanquantile(IRAFs, 0.96)
        r, p = stats.spearmanr(IRAFs, feature2vecs[feature],
                               nan_policy='omit')
        rs.append(r)
        print(f'\n- {feature}/{target_item} ({r=:.2f}, {p=:.3f}) -')


        feat_M = np.nanmean(feature2vecs[feature])
        for item, IRAF in zip(items, IRAFs):
            if np.isnan(IRAF): continue
            if IRAF < IRAF_cutoff: continue
            is_alike = feature2vecs[feature][list(items).index(item)]
            print(f'{item}: {IRAF=:.1f} ({is_alike}) ({feat_M:.1%})')
    return rs

@njit(fastmath=True, cache=True)
def numba_corr_by_context(a_mat, c_b_mat):
    num_items = a_mat.shape[0]
    num_columns = a_mat.shape[1]

    out = np.empty((num_items, num_items))
    for i_context in range(num_items):
        for j_item in range(num_items):
            total = 0
            cnt = 0
            for k in range(num_columns):
                if (np.isnan(a_mat[i_context, k]) or
                        np.isnan(c_b_mat[i_context, j_item, k])):
                    continue
                total += (a_mat[i_context, k] *
                          c_b_mat[i_context, j_item, k])
                cnt += 1
            if cnt == 0:
                out[i_context, j_item] = np.nan
            else:
                out[i_context, j_item] = total / cnt
    return out

@njit(fastmath=True, cache=True)
def numba_corr(a_mat, b_mat):
    num_items = a_mat.shape[0]
    num_columns = a_mat.shape[1]

    out = np.empty((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            total = 0
            cnt = 0
            for k in range(num_columns):
                if np.isnan(a_mat[i, k]) or np.isnan(b_mat[j, k]):
                    continue
                # if nan_mask[i, k] or nan_mask[j, k]:
                #     continue
                total += a_mat[i, k] * b_mat[j, k]
                cnt += 1
            if cnt == 0:
                out[i, j] = np.nan
            else:
                out[i, j] = total / cnt
    return out


if __name__ == '__main__':
    # it seems like functions emerge from pairing with alike (MR = .25) sometimes dissimilar
    #   perceptual has that much less (MR = .1) as does encyclopedic
    #   taxonomic is very dependent on alike (MR = .4) but never has dissimilar

    #   this doesn't apply to perceptual

    ITEMS = ['cow', 'giraffe', 'lobster', 'peach', 'pineapple', 'rabbit', 'raspberry',
             'salmon', 'shrimp', 'strawberry', 'tiger', 'tomato']
    # ITEMS = []

    ITEMS, _ = get_standard_items_list(pf_thresh=300)
    ITEMS = ITEMS[1:] # ambulence errors

    RS_ALL = []
    RS_ALL_ABS = []
    FEATURE_TYPE = 'functional'
    FEATURE_TYPE = 'encyclopedic'
    FEATURE_TYPE = 'taxonomic'
    FEATURE_TYPE = 'visual perceptual'

    # ITEMS = ['lobster']
    # ITEMS = ['helicopter', 'dove', 'armor']

    for ITEM in ITEMS:
        RS = find_context_helper(target_item=ITEM, feature_type=FEATURE_TYPE)
        RS_ABS = np.abs(np.array(RS))
        M_R = np.nanmean(RS)
        if np.isnan(M_R):
            print(f'Mystery NaN: {ITEM}')
            continue
        M_R_ABS = np.nanmean(RS_ABS)
        print(f'{ITEM} ({FEATURE_TYPE}): {M_R=:.3f}')
        RS_ALL.append(M_R)
        RS_ALL_ABS.append(M_R_ABS)
        if len(RS_ALL) > 2:
            RS_M_ALL = np.nanmean(RS_ALL)
            RS_SE_ALL = stats.sem(RS_ALL, nan_policy='omit')
            print(f'\t{FEATURE_TYPE}: M = {RS_M_ALL:.2f} [{RS_SE_ALL:.3f}]')
            RS_M_ALL = np.nanmean(RS_ALL_ABS)
            RS_SE_ALL = stats.sem(RS_ALL_ABS, nan_policy='omit')
            print(f'\t{FEATURE_TYPE}: M ABS = {RS_M_ALL:.2f} [{RS_SE_ALL:.3f}]')

