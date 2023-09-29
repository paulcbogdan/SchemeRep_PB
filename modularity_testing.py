import pickle
import numpy as np
from nichord import plot_chord
from nichord.combine import plot_and_combine
from pathlib import Path

from tqdm import tqdm

from atlas_utils import get_atlas
from plot_gen import plot_connectivity
from single_trial_conn import corr_matrix_last_two_dim, corr_last_dim
from utils import get_cache_RSA_fp, stdize, ndim_tril_flatten
from communities.algorithms import louvain_method
import networkx as nx
import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice.')

def org_activity_by_cin(ar_cin, activity):
    assert len(ar_cin) == activity.shape[0], \
        f'{ar_cin.shape=}, {activity.shape=}'
    new_shapes = (activity.shape[0], activity.shape[1], activity.shape[2]  // 3)
    cin2act = {1: np.full(new_shapes, np.nan),
               2: np.full(new_shapes, np.nan),
               3: np.full(new_shapes, np.nan)}
    for i, cin in enumerate(ar_cin):
        for j in range(1, 4):
            # strange that this needs to be transposed
            cin2act[j][i] = activity[i, :, cin == j].T
    return cin2act

# TODO: incorporate convert_matrix update to next version of nichord

def get_modules(corr, threshold_tile=.95, bonus_str=''):
    threshold = np.nanquantile(corr, threshold_tile)
    corr_thresh = corr.copy()
    corr_thresh[corr_thresh < threshold] = 0
    corr_thresh[corr_thresh >= threshold] = 1
    corr[corr < threshold] = 0
    import leidenalg
    import igraph as ig
    g = ig.Graph.Weighted_Adjacency(corr_thresh)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    return part
    #
    # for i, p in enumerate(part):
    #     print(f'{p=}')
    #     if len(p) == 1:
    #         continue
    #     plot_nichord(get_partition_matrix(corr, p), coords,
    #                  title=f'{bonus_str} Module {i}')

def get_partition_matrix(mat, idx, w_zeros=False):
    if w_zeros:
        mat_new = np.zeros(mat.shape)
        mat_new[np.ix_(idx, idx)] = mat[np.ix_(idx, idx)]
        return mat_new
    else:
        meshy = np.ix_(idx, idx)
        slicer = tuple([slice(None)] * (mat.ndim - 2) + [meshy[0], meshy[1]])
        # print(meshy)
        # quit()
        return mat[slicer]
        # slicer = tuple([slice(None)] * (ar.ndim - 2) + [tril[0], tril[1]])
        # return mat[np.ix_(idx, idx)]
def plot_nichord(corr, coords, fn, title, dir_out='nichord_plots'):
    from nichord.convert import convert_matrix
    from nichord.coord_labeler import get_idx_to_label

    edges, edge_weights = convert_matrix(corr)
    idx_to_label = get_idx_to_label(coords, atlas='yeo')

    network_colors = {'Uncertain': 'black', 'Visual': 'purple',
                      'SM': 'darkturquoise', 'DAN': 'green', 'VAN': 'fuchsia',
                      'Limbic': 'burlywood', 'FPCN': 'orange', 'DMN': 'red'}

    network_order = ['FPCN', 'DMN', 'DAN', 'Visual', 'SM', 'Limbic',
                     'Uncertain', 'VAN']
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    plot_and_combine(dir_out, fn, idx_to_label, edges,
                     edge_weights=edge_weights, coords=coords,
                     network_order=network_order, network_colors=network_colors,
                     title=title, chord_kwargs={'alphas': .5})

def get_conn_mat(atlas, fp_fMRI_col = 'obj_fMRI',
                 age=1, early=True, semantic=False, cin=None,
                 bilateral=False, combine_regions=False, vec_prod=False,
                 org_by_region=False, rxr=False):

    fp = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, DNN_layer=2,
                          fp_fMRI_col=fp_fMRI_col,
                          bilateral=bilateral, combine_regions=combine_regions,
                          vec_prod=vec_prod, org_by_region=org_by_region,
                          rxr=rxr)
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

    ar_cin = np.array(d['bhv']['CIN'])
    cin2act = org_activity_by_cin(ar_cin, activity)
    cin2corr = {}
    for cin, act in cin2act.items():
        cin2corr[cin], _ = corr_matrix_last_two_dim(act)

    corr_all, _ = corr_matrix_last_two_dim(activity)
    # corr_all_M = np.nanmean(corr_all, axis=0)
    return corr_all, cin2corr, atlas


def stuff():
    cin2corr_dif = {}
    for cin0, corr0 in cin2corr.items():
        for cin1, corr1 in cin2corr.items():
            if cin0 >= cin1:
                continue
            cin2corr_dif[(cin0, cin1)] = corr0 - corr1

    cin2corr_dif_M = {}
    cin2corr_t = {}
    for comparison, corr_dif in cin2corr_dif.items():
        cin2corr_dif_M[comparison] = np.mean(corr_dif, axis=0)
        SE = np.nanstd(corr_dif, axis=0) / np.sqrt(corr_dif.shape[0])
        cin2corr_t[comparison] = cin2corr_dif_M[comparison] / SE

    corr_all, _ = corr_matrix_last_two_dim(activity)
    corr_all_M = np.nanmean(corr_all, axis=0)
    corr_all_M[np.diag_indices_from(corr_all_M)] = np.nan
    partitions = get_modules(corr_all_M)

    for i, p in tqdm(enumerate(partitions), desc='looping partitions'):
        tmat = cin2corr_t[(1, 2)]
        pmat_t = get_partition_matrix(tmat, p)
        dir_out = fr'nichord_plots/cin12/{fp_fMRI_col}'
        fn = f'p{i}_tdif.png'
        title = f't-map. Cin 12, {fp_fMRI_col}, partition: {i}'
        plot_nichord(pmat_t, atlas['coords'], fn, title, dir_out=dir_out)

        fn = f'p{i}_M.png'
        title = f'mean. Cin 12, {fp_fMRI_col}, partition: {i}'
        pmat_M = get_partition_matrix(corr_all_M, p)
        plot_nichord(pmat_M, atlas['coords'], fn, title, dir_out=dir_out)



    # atlas['coords'], bonus_str=fp_fMRI_col


def compare_cross_corr(combine_regions=False, bilateral=False):
    atlas = get_atlas(combine_regions=combine_regions, bilateral=bilateral)

    fp2corr_all = {}
    fp2cin2corr = {}
    key0 = 'obj'
    key1 = 'vis'
    fps = [f'{key0}_fMRI', f'{key1}_fMRI']
    for fp in tqdm(fps, desc='prep conn'):
        name = fp.split('_')[0]
        fp2corr_all[name], fp2cin2corr[name], _ = get_conn_mat(atlas,
                                                               fp_fMRI_col=fp)

    corr_all_all = np.concatenate([fp2corr_all[key0], fp2corr_all[key1]], axis=0)
    corr_all_M = np.nanmean(corr_all_all, axis=0)
    partitions = get_modules(corr_all_M)
    for i, p in enumerate(partitions):
        cin2rs = {}
        if len(p) < 3:
            # print(f'Partition too small: {i}')
            continue
        # print(f'{len(p)=}')
        for cin in [1, 2]:
            # print(fp2cin2corr['obj'][cin].shape)
            obj_corr = get_partition_matrix(fp2cin2corr[key0][cin], p)
            # print(obj_corr.shape)
            obj_flat = ndim_tril_flatten(obj_corr)
            scn_corr = get_partition_matrix(fp2cin2corr[key1][cin], p)
            scn_flat = ndim_tril_flatten(scn_corr)
            rs = corr_last_dim(obj_flat, scn_flat)
            r_m = np.mean(rs)
            r_se = np.std(rs) / np.sqrt(len(rs))
            print(f'Partition {i} [len = {len(p)}]: '
                  f'{cin=}, {r_m=:.5f}, {r_se=:.5f}')
            cin2rs[cin] = rs
        t, p = stats.ttest_rel(cin2rs[1], cin2rs[2])
        print(f'\tDif ({i}) | {t=:.2f}, {p=:.3f}')

def make_basic_corr_comparison(combine_regions=True, bilateral=False):
    atlas = get_atlas(combine_regions=combine_regions,
                      bilateral=bilateral)

    fp2corr_all = {}
    fp2cin2corr = {}
    keys = ['obj', 'scn', 'con', 'vis']
    # keys = ['con']
    fps = [f'{key}_fMRI' for key in keys]
    # key0 = 'obj'
    # key1 = 'scn'
    # fps = [f'{key0}_fMRI', f'{key1}_fMRI']
    for fp in tqdm(fps, desc='prep conn'):
        name = fp.split('_')[0]
        fp2corr_all[name], fp2cin2corr[name], _ = \
            get_conn_mat(atlas, combine_regions=combine_regions,
                         bilateral=bilateral, fp_fMRI_col=fp)
        fp2corr_all[name] = fp2corr_all[name][:3]


    for name, cin2corr in fp2cin2corr.items():
        corr_c = cin2corr[1]
        corr_i = cin2corr[2]
        corr_dif = corr_c - corr_i
        corr_dif_M = np.nanmean(corr_dif, axis=0)
        corr_dif_M[np.diag_indices_from(corr_dif_M)] = np.nan
        corr_dif_SD = np.nanstd(corr_dif, axis=0)
        corr_dif_N = np.sum(~np.isnan(corr_dif), axis=0)
        corr_dif_SE = corr_dif_SD / np.sqrt(corr_dif_N)
        t = corr_dif_M / corr_dif_SE
        # t = corr_c[1]
        # t = np.nanmean(corr_c, axis=0)
        title = f'{name}, Cin 1 - Cin 2'
        plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'], title=title, no_avg=True,
                      cbar_label='t-value')
        # quit()



if __name__  == '__main__':
    compare_cross_corr()
