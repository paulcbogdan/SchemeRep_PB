from atlas_utils import get_atlas
from network_clf import generic_prep
from utils import stdize
import numpy as np

np.random.seed(0)

def ar2conn(ar, flat=True):
    ar = stdize(ar, axis=2, nans=True)
    conn = ar[..., None, :] * ar[..., None, :, :]
    conn = np.nanmean(conn, axis=-1)
    k = conn.shape[-1]
    if flat:
        idxs = np.tril_indices(k, k=1)
        conn = conn[..., idxs[0], idxs[1]]
    return conn

def stratify(ar10, ar11):
    for sn in range(ar10.shape[0]):
        ar10_sn = ar10[sn]
        nan_trials0 = np.all(np.isnan(ar10_sn), axis=0)
        n_nan0 = np.sum(nan_trials0)
        n_good0 = ar10_sn.shape[-1] - n_nan0
        ar11_sn = ar11[sn]
        nan_trials1 = np.all(np.isnan(ar11_sn), axis=0)
        n_nan1 = np.sum(nan_trials1)
        n_good1 = ar11_sn.shape[-1] - n_nan1
        dif = n_good0 - n_good1
        # print(f'{sn=}, {n_good0=}, {n_good1=}: {dif=}')
        if dif > 0:
            non_nan_trials0 = ~nan_trials0
            non_nan_trials0_idxs = np.argwhere(non_nan_trials0).flatten()
            non_nan_trials0_idxs = np.random.choice(non_nan_trials0_idxs,
                                                    size=dif, replace=False)
            ar10[sn, :, non_nan_trials0_idxs] = np.nan
        elif dif < 0:
            non_nan_trials1 = ~nan_trials1
            non_nan_trials1_idxs = np.argwhere(non_nan_trials1).flatten()
            non_nan_trials1_idxs = np.random.choice(non_nan_trials1_idxs,
                                                    size=-dif, replace=False)
            ar11[sn, :, non_nan_trials1_idxs] = np.nan
    return ar10, ar11



def similarity_analysis(age2idxs, ar00, ar01, ar10, ar11):
    ar10_l = []
    ar11_l = []
    for i in range(10):
        ar10_, ar11_ = stratify(ar10.copy(), ar11.copy())
        ar10_l.append(ar10_)
        ar11_l.append(ar11_)
    ar10 = np.concatenate(ar10_l, axis=-1)
    ar11 = np.concatenate(ar11_l, axis=-1)
    print('Finished stratifying')
    conn00 = ar2conn(ar00)
    conn01 = ar2conn(ar01)
    conn10 = ar2conn(ar10)
    conn11 = ar2conn(ar11)
    print('Finished making connectomes')

    for age in [1, 2]:
        idxs = age2idxs[age]
        conn00_ = conn00[idxs]
        conn01_ = conn01[idxs]
        conn10_ = conn10[idxs]
        conn11_ = conn11[idxs]
        file0 = np.array([conn00_, conn01_])
        file0 = stdize(file0, axis=2, nans=True)
        file1 = np.array([conn10_, conn11_])
        file1 = stdize(file1, axis=2, nans=True)
        similarity_mats = file0[:, None, ...] * file1[None, ...]
        similarity_mats = np.nanmean(similarity_mats, axis=-1)
        for cond0 in [0, 1]:
            for cond1 in [0, 1]:
                M = np.nanmean(similarity_mats[cond0, cond1])
                SD = np.nanstd(similarity_mats[cond0, cond1])
                N = np.sum(~np.isnan(similarity_mats[cond0, cond1]))
                SE = SD / np.sqrt(N)
                print(f'{age=}, (cond {cond0}x{cond1}): {M:.3f} +/- {SE:.3f}')
        itr = similarity_mats[0, 0] - similarity_mats[0, 1] - \
                similarity_mats[1, 0] + similarity_mats[1, 1]
        itr_M = np.nanmean(itr)
        itr_SD = np.nanstd(itr)
        itr_N = np.sum(~np.isnan(itr))
        itr_SE = itr_SD / np.sqrt(itr_N)
        itr_t = itr_M / itr_SE
        print(f'{age=}, (itr): {itr_M:.3f} +/- {itr_SE:.3f} | {itr_t=:.2f}')
        con_t = similarity_mats[1, 0] - similarity_mats[1, 1]
        con_M = np.nanmean(con_t)
        con_SD = np.nanstd(con_t)
        con_N = np.sum(~np.isnan(con_t))
        con_SE = con_SD / np.sqrt(con_N)
        con_t = con_M / con_SE
        print(f'{age=}, (con): {con_M:.3f} +/- {con_SE:.3f} | {con_t=:.2f}')


def do_similarity_analysis(threshold=0.95):
    fp = 'obj7_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'odd_even': False,
              'combine_regions': False,
              }
    partitions, sn_inc_activity0, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)

    fp = 'obj7_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'hit_hit',
              'atlas_name': 'BNA',
              # 'key_vals': ('30', '31'),
              'key_vals': (False, True),
              'combine_regions': False,
              }
    partitions, sn_inc_activity1, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)

    atlas = get_atlas(combine_regions=kwargs['combine_regions'])
    keep_regions = ['IPL', 'MFG', 'IFG', 'SFG']
    # keep_regions = ['LOC', 'sOcG', 'EVC', 'FuG', 'ATL']
    # keep_regions = ['LOC', 'sOcG', 'EVC', 'FuG', 'ATL',
    #                 'IPL', 'MFG', 'IFG', 'SFG']
    keep_idxs = [i for region in keep_regions
                 for (i, name) in enumerate(atlas['ROIs'])
                 if region in name]
    sn_inc_activity0 = sn_inc_activity0[:, :, keep_idxs]
    sn_inc_activity1 = sn_inc_activity1[:, :, keep_idxs]

    similarity_analysis(age2idxs, sn_inc_activity0[:, 0],
                        sn_inc_activity0[:, 1],
                        sn_inc_activity1[:, 0],
                        sn_inc_activity1[:, 1])


if __name__ == '__main__':
    # a = np.array([10, 20])
    # b = np.array([1, 2])
    # c = a[:, None, ...] + b[None, ...]
    # print(c)
    # print(c[0, 1])
    # quit()
    do_similarity_analysis()
