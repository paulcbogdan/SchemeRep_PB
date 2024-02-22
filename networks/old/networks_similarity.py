from connsearch import print_list_stats

from atlas_utils import get_atlas
from network_clf import generic_prep
from ttest_mat import get_stats_graphs
from networks.old.network_funcs import load_FC_for_Lifu
from utils import stdize, pickle_wrap
import numpy as np

np.random.seed(0)

def ar2conn(ar, flat=True, mask=None):
    ar = stdize(ar, axis=2, nans=True)
    conn = ar[..., None, :] * ar[..., None, :, :]
    conn = np.nanmean(conn, axis=-1)
    if mask is not None:
        conn[..., ~mask] = np.nan
    k = conn.shape[-1]
    if flat:
        idxs = np.tril_indices(k, k=-1)
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

def stratify2(ar00, ar01, ar10, ar11):
    ars = [ar00, ar01, ar10, ar11]
    for sn in range(ar00.shape[0]):
        ar_goods = []
        for i, ar in enumerate(ars):
            ar_sn = ar[sn]
            nan_trials = np.all(np.isnan(ar_sn), axis=0)
            goods = np.argwhere(~nan_trials).flatten()
            ar_goods.append(goods)

        intersect00 = np.intersect1d(ar_goods[0], ar_goods[2])
        intersect01 = np.intersect1d(ar_goods[0], ar_goods[3])
        dif = len(intersect00) - len(intersect01)
        # print(f'{sn=}, first {dif=}')
        if dif > 0:
            intersect00 = np.random.choice(intersect00, size=dif,
                                           replace=False)
            ar00[sn, :, intersect00] = np.nan
        elif dif < 0:
            intersect01 = np.random.choice(intersect01, size=-dif,
                                           replace=False)
            ar01[sn, :, intersect01] = np.nan
        intersect10 = np.intersect1d(ar_goods[1], ar_goods[2])
        intersect11 = np.intersect1d(ar_goods[1], ar_goods[3])
        dif = len(intersect10) - len(intersect11)
        if dif > 0:
            intersect10 = np.random.choice(intersect10, size=dif,
                                           replace=False)
            ar10[sn, :, intersect10] = np.nan
        elif dif < 0:
            intersect11 = np.random.choice(intersect11, size=-dif,
                                           replace=False)
            ar11[sn, :, intersect11] = np.nan
    return ar00, ar01, ar10, ar11

def similarity_analysis(age2idxs, ar00, ar01, ar10, ar11, mask=None):
    # ar00_l = []
    # ar01_l = []
    ar10_l = []
    ar11_l = []
    for i in range(10):
        ar10_, ar11_ = stratify(ar10.copy(), ar11.copy())
    #     # ar00_, ar01_, ar10_, ar11_ = stratify2(ar00.copy(), ar01.copy(),
    #     #                                        ar10.copy(), ar11.copy())
    #     ar00_l.append(ar00_)
    #     ar01_l.append(ar01_)
        ar10_l.append(ar10_)
        ar11_l.append(ar11_)
    # ar00 = np.concatenate(ar00_l, axis=-1)
    # ar01 = np.concatenate(ar01_l, axis=-1)
    # ar10 = np.concatenate(ar10_l, axis=-1)
    # ar11 = np.concatenate(ar11_l, axis=-1)
    # print('Finished stratifying')
    conn00 = ar2conn(ar00, mask=mask)
    # print(f'{ar00.shape=}')
    # print(f'{conn00.shape=}')
    # quit()
    conn01 = ar2conn(ar01, mask=mask)
    # conn00 = np.nanmean(conn00, axis=1)
    # conn01 = np.nanmean(conn01, axis=1)

    # print(f'{conn00.shape=}')
    # print(ar10.shape)
    conn10 = ar2conn(ar10, mask=mask)
    # conn10 = np.nanmean(conn10, axis=1)
    # conn10 = conn10[:, 1]
    conn11 = ar2conn(ar11, mask=mask)
    # conn11 = np.nanmean(conn11, axis=1)
    # conn11 = conn11[:, 1]

    # conn00 = ar2conn(ar00[..., :57])
    # conn01 = ar2conn(ar01[..., :57])
    # conn10 = ar2conn(ar10[..., 57:])
    # conn11 = ar2conn(ar11[..., 57:])
    print('Finished making connectomes')

    ts = []
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
        similarity_mats = np.arctanh(similarity_mats)

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
        ts.append(itr_t)
    return ts

def get_p_mask(kwargs):
    kwargs = kwargs.copy()
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='../cache')
    M2_graph, SD2_graph, SE2_graph, N2_graph, t2_graph, p2_graph, z2_graph = \
        get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                         sn_inc_conn[age2idxs[2], 1, :, :])
    return p2_graph < 0.05

def do_similarity_analysis(threshold=0.95):
    fp = 'obj7_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'odd_even': False,
              'combine_regions': False,
              }
    mask = get_p_mask(kwargs)

    partitions, sn_inc_activity0, age2idxs, _, i2name = \
        generic_prep(kwargs, threshold=threshold)



    fp = 'obj7_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc_hit_hit',
              'atlas_name': 'BNA',
              'key_vals': ('10', '20', '30',
                           '11', '21', '31'),
              # 'key_vals': (False, True),
              'combine_regions': False,
              }
    # fp = 'obj7_fMRI'
    # kwargs = {'fp': fp,
    #           'split': False,
    #           'key': 'inc_hit_hit',
    #           'atlas_name': 'BNA',
    #           'key_vals': ('20', '21'),
    #           'combine_regions': False,
    #           }
    partitions, sn_inc_activity1, age2idxs, _, i2name = \
        generic_prep(kwargs, threshold=threshold)


    # print(sn_inc_activity1.shape)
    # for i in range(6):
    #     n_non_nans = np.sum(~np.isnan(sn_inc_activity1[:, i, 0, :]))
    #     print(f'{i}: {n_non_nans / 114:.3f}')
    # quit()

    atlas = get_atlas(combine_regions=kwargs['combine_regions'])
    rois = atlas['ROIs']
    keep_idxs = list(range(len(rois)))
    #
    # keep_regions = ['IPL', 'pSTS', 'MFG', 'IFG', 'SFG']
    keep_regions = atlas['tick_labels']
    bad_regions = ['Tha', 'Str']
    # keep_regions = ['LOC', 'sOcG', 'EVC', 'FuG', 'ATL']
    # keep_regions = ['LOC', 'sOcG', 'EVC', 'FuG', 'ATL',
    #                 'IPL', 'MFG', 'IFG', 'SFG']
    # print(f'{len(keep_idxs)=}')
    for idx in list(keep_idxs):
        name = rois[idx]
        if not any(region in name for region in keep_regions):
            keep_idxs.remove(idx)
            continue
        if any(region in name for region in bad_regions):
            keep_idxs.remove(idx)
            continue
    keep_idxs = sorted(list(set(keep_idxs)))
    mask = mask[keep_idxs, :][:, keep_idxs]

    sn_inc_activity0 = sn_inc_activity0[:, :, keep_idxs]
    sn_inc_activity1 = sn_inc_activity1[:, :, keep_idxs]
    # similarity_analysis(age2idxs, sn_inc_activity0[:, 0],
    #                                  sn_inc_activity0[:, 1],
    #                                  sn_inc_activity1[:, 0],
    #                                  sn_inc_activity1[:, 1],
    #                     mask=None)
    t_YA, t_OA = similarity_analysis(age2idxs, sn_inc_activity0[:, 0],
                                     sn_inc_activity0[:, 1],
                                     sn_inc_activity1[:, 1],
                                     sn_inc_activity1[:, 4],
                                     mask=mask)
    print('----------------------')
    quit()
    ts_YA = []
    ts_OA = []
    for _ in range(100):
        sn_inc_activity0, sn_inc_activity1 = \
            shuffle_similarity_analysis(sn_inc_activity0, sn_inc_activity1)
        t_YA, t_OA = similarity_analysis(age2idxs, sn_inc_activity0[:, 0],
                            sn_inc_activity0[:, 1],
                            sn_inc_activity1[:, 0],
                            sn_inc_activity1[:, 1],
                                         mask=None)
        # t_YA, t_OA = similarity_analysis(age2idxs, sn_inc_activity0[:, 0],
        #                     sn_inc_activity0[:, 1],
        #                     sn_inc_activity1[:, 1],
        #                     sn_inc_activity1[:, 4],)
        ts_YA.append(t_YA)
        ts_OA.append(t_OA)
        print_list_stats(ts_OA)
    # similarity_analysis(age2idxs, sn_inc_activity0[:, 0],
    #                     sn_inc_activity0[:, 1],
    #                     sn_inc_activity1[:, :3],
    #                     sn_inc_activity1[:, 3:])
    # similarity_analysis(age2idxs, sn_inc_activity1[:, [0, 3]],
    #                     sn_inc_activity1[:, [2, 5]],
    #                     sn_inc_activity1[:, :3],
    #                     sn_inc_activity1[:, 3:])

def shuffle_similarity_analysis(ar0, ar1):
    rng = np.random.default_rng()
    ar_both = np.concatenate([ar0, ar1], axis=1)
    ar_combined = np.nanmean(ar_both, axis=1)
    rng.shuffle(ar_combined, axis=-1)
    ar_both = np.transpose(ar_both, (1, 0, 2, 3))
    for cond in range(ar_both.shape[0]):
        non_nan = ~np.isnan(ar_both[cond])
        # print(ar_combined.shape)
        ar_both[cond, non_nan] = ar_combined[non_nan]
    ar_both = np.transpose(ar_both, (1, 0, 2, 3))
    ar0 = ar_both[:, :ar0.shape[1]]
    ar1 = ar_both[:, ar0.shape[1]:]
    return ar0, ar1



if __name__ == '__main__':
    # a = np.array([10, 20])
    # b = np.array([1, 2])
    # c = a[:, None, ...] + b[None, ...]
    # print(c)
    # print(c[0, 1])
    # quit()
    do_similarity_analysis()
