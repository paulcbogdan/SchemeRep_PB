import pickle
import numpy as np
from nichord.combine import plot_and_combine
from pathlib import Path

from tqdm import tqdm

from atlas_utils import get_atlas
from old.plot_gen import plot_connectivity
from utils import get_RSA_fn, tril_flat
import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice.')

def org_activity_by_cin(ar_inc, activity):
    assert len(ar_inc) == activity.shape[0], \
        f'{ar_inc.shape=}, {activity.shape=}'
    new_shapes = (activity.shape[0], activity.shape[1], activity.shape[2]  // 3)
    inc2act = {1: np.full(new_shapes, np.nan),
               2: np.full(new_shapes, np.nan),
               3: np.full(new_shapes, np.nan)}
    for i, inc in enumerate(ar_inc):
        for j in range(1, 4):
            # strange that this needs to be transposed
            inc2act[j][i] = activity[i, :, inc == j].T
    return inc2act

# TODO: incorporate convert_matrix update to next version of nichord

def get_modules(corr, threshold_tile=.9, bonus_str=''):
    threshold = np.nanquantile(corr, threshold_tile)
    corr_thresh = corr.copy()
    corr_thresh[corr_thresh < threshold] = 0
    corr_thresh[corr_thresh >= threshold] = 1
    corr[corr < threshold] = 0
    top_edges = corr > threshold
    import leidenalg
    import igraph as ig
    g = ig.Graph.Weighted_Adjacency(corr_thresh)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    return part, top_edges
    #
    # for i, p in enumerate(part):
    #     print(f'{p=}')
    #     if len(p) == 1:
    #         continue
    #     plot_nichord(get_partition_matrix(corr, p), coords,
    #                  title=f'{bonus_str} Module {i}')



def get_conn_mat(atlas, fp_fMRI_col = 'obj_fMRI',
                 age=1, early=True, semantic=False, inc=None,
                 bilateral=False, combine_regions=False, vec_prod=False,
                 org_by_region=False, rxr=False):

    fp = get_RSA_fn(inc=inc, age=age, semantic=semantic, DNN_layer=2,
                    fp_fMRI_col=fp_fMRI_col,
                    bilateral=bilateral, combine_regions=combine_regions,
                    vec_prod=vec_prod, org_by_region=org_by_region,
                    )
    with open(fp, 'rb') as file:
        d = pickle.load(file)

    activity = [np.array(d['activity'][roi1]) for roi1 in atlas['ROIs']]
    activity = np.array(activity)
    activity = activity.transpose((1, 0, 2))
    # activity = activity[0]
    # print(activity.shape)
    # corr = np.corrcoef(activity)
    # plot_connectivity(corr, atlas['ticks'], atlas['tick_labels'],
    #                   atlas['tick_lows'], no_avg=True, title=fp_fMRI_col + '_4')
    #
    #
    # quit()
    # for j in range(activity.shape[0]):
    #     idxs = np.arange(activity.shape[2])
    #     np.random.shuffle(idxs)
    #     activity[j, :, :] = activity[j, :, idxs].T
    # print(activity.shape)
    # quit()

    ar_inc = np.array(d['bhv']['CIN'])
    inc2act = org_activity_by_cin(ar_inc, activity)
    inc2corr = {}
    for inc, act in inc2act.items():
        inc2corr[inc], _ = corr_matrix_last_two_dim(act)

    corr_all, _ = corr_matrix_last_two_dim(activity)
    # corr_all_M = np.nanmean(corr_all, axis=0)
    return corr_all, inc2corr, atlas


def stuff():
    inc2corr_dif = {}
    for inc0, corr0 in inc2corr.items():
        for inc1, corr1 in inc2corr.items():
            if inc0 >= inc1:
                continue
            inc2corr_dif[(inc0, inc1)] = corr0 - corr1

    inc2corr_dif_M = {}
    inc2corr_t = {}
    for comparison, corr_dif in inc2corr_dif.items():
        inc2corr_dif_M[comparison] = np.mean(corr_dif, axis=0)
        SE = np.nanstd(corr_dif, axis=0) / np.sqrt(corr_dif.shape[0])
        inc2corr_t[comparison] = inc2corr_dif_M[comparison] / SE

    corr_all, _ = corr_matrix_last_two_dim(activity)
    corr_all_M = np.nanmean(corr_all, axis=0)
    corr_all_M[np.diag_indices_from(corr_all_M)] = np.nan
    partitions = get_modules(corr_all_M)

    for i, p in tqdm(enumerate(partitions), desc='looping partitions'):
        tmat = inc2corr_t[(1, 2)]
        pmat_t = get_partition_matrix(tmat, p)
        dir_out = fr'nichord_plots/inc12/{fp_fMRI_col}'
        fn = f'p{i}_tdif.png'
        title = f't-map. inc 12, {fp_fMRI_col}, partition: {i}'
        plot_nichord(pmat_t, atlas['coords'], fn, title, dir_out=dir_out)

        fn = f'p{i}_M.png'
        title = f'mean. inc 12, {fp_fMRI_col}, partition: {i}'
        pmat_M = get_partition_matrix(corr_all_M, p)
        plot_nichord(pmat_M, atlas['coords'], fn, title, dir_out=dir_out)



    # atlas['coords'], bonus_str=fp_fMRI_col


def compare_cross_corr(combine_regions=False, bilateral=False):
    atlas = get_atlas(combine_regions=combine_regions, combine_bilateral=bilateral)

    fp2corr_all = {}
    fp2inc2corr = {}
    key0 = 'obj'
    key1 = 'vis'
    fps = [f'{key0}_fMRI', f'{key1}_fMRI']
    for fp in tqdm(fps, desc='prep conn'):
        name = fp.split('_')[0]
        fp2corr_all[name], fp2inc2corr[name], _ = get_conn_mat(atlas,
                                                               fp_fMRI_col=fp)

    corr_all_all = np.concatenate([fp2corr_all[key0], fp2corr_all[key1]], axis=0)
    corr_all_M = np.nanmean(corr_all_all, axis=0)
    partitions = get_modules(corr_all_M)
    for i, p in enumerate(partitions):
        inc2rs = {}
        if len(p) < 3:
            # print(f'Partition too small: {i}')
            continue
        # print(f'{len(p)=}')
        for inc in [1, 2]:
            # print(fp2inc2corr['obj'][inc].shape)
            obj_corr = get_partition_matrix(fp2inc2corr[key0][inc], p)
            # print(obj_corr.shape)
            obj_flat = tril_flat(obj_corr)
            scn_corr = get_partition_matrix(fp2inc2corr[key1][inc], p)
            scn_flat = tril_flat(scn_corr)
            rs = corr_last_dim(obj_flat, scn_flat)
            r_m = np.mean(rs)
            r_se = np.std(rs) / np.sqrt(len(rs))
            print(f'Partition {i} [len = {len(p)}]: '
                  f'{inc=}, {r_m=:.5f}, {r_se=:.5f}')
            inc2rs[inc] = rs
        t, p = stats.ttest_rel(inc2rs[1], inc2rs[2])
        print(f'\tDif ({i}) | {t=:.2f}, {p=:.3f}')

def make_basic_corr_comparison(combine_regions=True, bilateral=False):
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=bilateral)

    fp2corr_all = {}
    fp2inc2corr = {}
    keys = ['obj', 'scn', 'con', 'vis']
    # keys = ['con']
    fps = [f'{key}_fMRI' for key in keys]
    # key0 = 'obj'
    # key1 = 'scn'
    # fps = [f'{key0}_fMRI', f'{key1}_fMRI']
    for fp in tqdm(fps, desc='prep conn'):
        name = fp.split('_')[0]
        fp2corr_all[name], fp2inc2corr[name], _ = \
            get_conn_mat(atlas, combine_regions=combine_regions,
                         bilateral=bilateral, fp_fMRI_col=fp)
        fp2corr_all[name] = fp2corr_all[name][:3]


    for name, inc2corr in fp2inc2corr.items():
        corr_c = inc2corr[1]
        corr_i = inc2corr[2]
        corr_dif = corr_c - corr_i
        corr_dif_M = np.nanmean(corr_dif, axis=0)
        corr_dif_M[np.diag_indices_from(corr_dif_M)] = np.nan
        corr_dif_SD = np.nanstd(corr_dif, axis=0)
        corr_dif_N = np.sum(~np.isnan(corr_dif), axis=0)
        corr_dif_SE = corr_dif_SD / np.sqrt(corr_dif_N)
        t = corr_dif_M / corr_dif_SE
        # t = corr_c[1]
        # t = np.nanmean(corr_c, axis=0)
        title = f'{name}, inc 1 - inc 2'
        plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'], title=title, no_avg=True,
                      cbar_label='t-value')
        # quit()



if __name__  == '__main__':
    compare_cross_corr()
