import pickle
import numpy as np

from DNN_vectors import get_DNN_vecs
from ROIs import get_atlas
from analyze_conn import bulk_correlate
from organize_bhv import get_trial_info
from plotting import plot_connectivity
from stim_vec import get_stim_RDM
from utils import get_cache_RSA_fp
from wordvec_get_vectors import get_semantic_vectors
import scipy.stats as stats

def roimap2np(ROI2ar, only_some=None):
    l = []
    for ROI, ar in ROI2ar.items():
        if only_some:
            for ROI_ in only_some:
                if ROI_ in ROI:
                    break
            else:
                continue
        l.append(ar)
    # print(np.array(l).shape)
    l = np.array(l)
    print(f'Data shape: {l.shape}')
    return l

def corr_matrix_last_two_dim(ar, nans=True):
    # Second-to-last dim should be your variable
    # Last dim should be a time series
    m = np.nanmean if nans else np.mean
    s = np.nanstd if nans else np.std
    # Creates nan in rs and rs_flat if every value in time series is identical
    M = np.expand_dims(m(ar, axis=-1), axis=-1)
    SD = np.expand_dims(s(ar, axis=-1), axis=-1)
    ar_std = (ar - M) / SD
    ar_std0 = np.expand_dims(ar_std, axis=-2)
    ar_std1 = np.expand_dims(ar_std, axis=-3)
    rs = ar_std0 * ar_std1
    rs = m(rs, axis=-1)
    tril = np.tril_indices(rs.shape[-1], k=-1)
    slicer = [slice(None)] * (rs.ndim - 2) + [tril[0], tril[1]]
    rs_flat = rs[slicer]
    return rs, rs_flat

def corr_last_dim(ar0, ar1, nans=True):
    # Last dim of both should be a time series
    m = np.nanmean if nans else np.mean
    s = np.nanstd if nans else np.std
    M0 = np.expand_dims(m(ar0, axis=-1), axis=-1)
    SD0 = np.expand_dims(s(ar0, axis=-1), axis=-1)
    ar0_ = (ar0 - M0) / SD0
    M1 = np.expand_dims(m(ar1, axis=-1), axis=-1)
    SD1 = np.expand_dims(s(ar1, axis=-1), axis=-1)
    ar1_ = (ar1 - M1) / SD1
    rs = ar0_ * ar1_
    rs = m(rs, axis=-1)
    return rs


def do_single_trial_conn(age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=False, vec_prod=False,
                 org_by_region=False, rxr=False):

    fp = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, DNN_layer=2,
                          fp_fMRI_col='obj_fMRI',
                          bilateral=bilateral, combine_regions=combine_regions,
                          vec_prod=vec_prod, org_by_region=org_by_region,
                          rxr=rxr)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    ROI_focus = ['EVC', 'LOC', 'sOcG']
    # ROI_focus = ['IFG', 'MFG']
    obj_a = roimap2np(d['activity'], only_some=ROI_focus)

    fp2 = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, DNN_layer=2,
                          fp_fMRI_col='scn_fMRI',
                          bilateral=bilateral, combine_regions=combine_regions,
                          vec_prod=vec_prod, org_by_region=org_by_region,
                          rxr=rxr)
    with open(fp2, 'rb') as file:
        d2 = pickle.load(file)
    scn_a = roimap2np(d2['activity'], only_some=ROI_focus)

    both_a = np.stack([obj_a, scn_a], axis=-1)
    both_a = both_a.transpose((1, 2, 0, 3))
    _, rs_flat = corr_matrix_last_two_dim(both_a)
    _, RDMs_flat = corr_matrix_last_two_dim(rs_flat)

    df_sn = get_trial_info('102')
    df_sn.sort_values(by='obj', inplace=True)
    # d_vecs = get_semantic_vectors()
    d_vecs = get_DNN_vecs(DNN_layer=-1, PCA=True)
    RDM_stim = get_stim_RDM(df_sn, d_vecs, add=True)
    print(f'stim: {np.sum(np.isnan(RDM_stim))}')
    RDM_stim_flat = RDM_stim[np.tril_indices(RDM_stim.shape[0], k=-1)]

    r2nd_order = corr_last_dim(RDMs_flat, RDM_stim_flat)
    r2nd_order = np.arctanh(r2nd_order)
    M_second_order = np.nanmean(r2nd_order, axis=0)
    SD_second_order = np.nanstd(r2nd_order, axis=0)
    SE_second_order = SD_second_order / np.sqrt(r2nd_order.shape[0])
    t = M_second_order / SE_second_order
    p = 2*(1 - stats.t.cdf(np.abs(t), r2nd_order.shape[0] - 1))
    print(f'{r2nd_order=}')
    print(f'{M_second_order=:.3f}')
    print(f'{SD_second_order=:.3f}')
    print(f'{t=:.3f}, {p=:.3f}')
    # print(np.argwhere(np.isnan(r2nd_order)))
    quit()


    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      bilateral=bilateral or org_by_region)
    title = 'sleepy'
    plot_connectivity(np.mean(rs, axis=0),
                      atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title=title, no_avg=True,
                      cbar_label='t-value')



if __name__ == '__main__':
    do_single_trial_conn()