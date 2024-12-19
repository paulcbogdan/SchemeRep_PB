import numpy as np

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_analysis import get_devereux_RSM_by_type
from llama.devereux_llama import get_standard_items_list
import scipy.stats as stats


def get_w2v_vec_deve(stim, w2v):
    parts = stim.split(' ')
    vecs = []
    for part in parts:
        try:
            vec = w2v[part]
            vecs.append(vec)
        except KeyError:
            print(f'Bad {stim}: {part}')
    return np.mean(vecs, axis=0)


def get_d_vecs_w2v_deve(pf_thresh=250, normalize=True):
    items = get_standard_items_list(pf_thresh)
    from gensim import downloader
    w2vectors = downloader.load('word2vec-google-news-300')
    d_vecs = {}
    item_remap = {'doughnut': 'donut',
                  'land_rover': 'landrover', # 'car' is already in the dataset
                  'range_rover': 'SUV'
                  }
    for item in items:
        if item in item_remap:
            item_ = item_remap[item]
        else:
            item_ = item
        vec = get_w2v_vec_deve(item_, w2vectors)
        # print(f'| {vec.shape=}')
        # d_vecs[(item, item, item)] = vec
        d_vecs[item] = vec
    if normalize:
        # idt this does anything to correlations because it averages within-vec
        vecs_all = [vec for vec in d_vecs.values()]
        # vecs_all = np.array(list(d_vecs.values()))
        vec_SDs = np.nanstd(vecs_all, axis=0)
        vec_Ms = np.nanmean(vecs_all, axis=0)
        for item, vec in d_vecs.items():
            d_vecs[item] = (vec - vec_Ms) / vec_SDs
    return d_vecs


def get_w2v_deve_RSM(pf_thresh=250):
    items = get_standard_items_list(pf_thresh)
    d_vecs = pickle_wrap(get_d_vecs_w2v_deve,
                         kwargs={'pf_thresh': pf_thresh})
    items_M_vecs = []
    for item in items:
        items_M_vecs.append(d_vecs[item])
    items_M_vecs = np.array(items_M_vecs)
    RSM = np.corrcoef(items_M_vecs)
    return RSM


def test_w2v_dev():
    RSM = get_w2v_deve_RSM(pf_thresh=250)
    trils = np.tril_indices_from(RSM, k=-1)
    RSM_flat = RSM[trils]

    type2RSM = get_devereux_RSM_by_type(attn=False)
    for feature_type, RSM_feat in type2RSM.items():
        assert RSM_feat.shape == RSM.shape, f'{RSM_feat.shape=} {RSM.shape=}'
        RSM_feat_flat = RSM_feat[trils]
        RSM_feat_nans = np.isnan(RSM_feat_flat)
        num_nans = np.sum(RSM_feat_nans)
        RSM_flat_ = RSM_flat[~RSM_feat_nans]
        RSM_feat_flat_ = RSM_feat_flat[~RSM_feat_nans]
        r, p = stats.spearmanr(RSM_flat_, RSM_feat_flat_,
                               nan_policy='omit' if num_nans > 0 else 'raise')

        prop_nan = num_nans / len(RSM_feat_flat)
        if prop_nan > 0:
            print(f'\t{feature_type=}: {r=:.3f}, {p=:.3f} ({prop_nan=:.1%})')
        else:
            print(f'\t{feature_type=}: {r=:.3f}, {p=:.3f}')


if __name__ == '__main__':
    # get_w2v_deve_RSM(pf_thresh=250)
    test_w2v_dev()