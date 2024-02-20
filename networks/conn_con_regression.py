import os

import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ttest_mat import get_stats_graphs

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

from scipy import stats

from atlas_utils import get_atlas
from old.modularity import get_BNA_coords, plot_nichord, get_main_partitions, get_partition_matrix
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from utils import pickle_wrap
from NBS import get_NBS_clusters
import numpy as np
import statsmodels.formula.api as smf

def combine_bl(sn_inc_conn):
    n_sn = sn_inc_conn.shape[0]
    n_cond = sn_inc_conn.shape[1]
    nroi = sn_inc_conn.shape[2]
    sn_inc_conn_mid = np.full((n_sn, n_cond, nroi // 2, nroi), np.nan)
    sn_inc_conn_new = np.full((n_sn, n_cond, nroi // 2, nroi // 2), np.nan)
    for i in range(nroi // 2):
        i_low = i * 2
        i_high = i_low + 2
        sn_inc_conn_mid[:, :, i, :] = (
            sn_inc_conn[:, :, i_low:i_high, :].mean(axis=2))
        # print(sn_inc_conn_mid[:, :, i, :])
        # quit()
    for i in range(nroi // 2):
        i_low = i * 2
        i_high = i_low + 2
        sn_inc_conn_new[:, :, :, i] = (
            sn_inc_conn_mid[:, :, :, i_low:i_high].mean(axis=3))
    return sn_inc_conn_new

def get_con_reg_zs(fp='obj7_fMRI', combine_regions=False, age='healthy',
                   comb_bl=True):
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              'combine_regions': combine_regions
              }

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    sn_inc_activity = np.nanmean(sn_inc_activity, axis=1)
    atlas = get_atlas(combine_regions=combine_regions)
    labels = atlas['ROI_regions']
    bad_labels = {'Str', 'Tha', 'Amyg'}
    for i, label in enumerate(labels):
        for bad_label in bad_labels:
            if bad_label in label:
                sn_inc_conn[:, :, i, :] = np.nan
                sn_inc_conn[:, :, :, i] = np.nan
    #     n_nans = np.sum(np.isnan(sn_inc_activity[:, i, :]))
    #     possible = sn_inc_activity.shape[0] * sn_inc_activity.shape[2]
    #     prop_nan = n_nans / possible
    #     print(f'{label}: {prop_nan:.3f}')
    #
    #
    #     # print(sn_inc_activity.shape)
    #     # quit()
    # quit()

    # if comb_bl:
    #     sn_inc_conn = combine_bl(sn_inc_conn)

    # sn_inc_conn[:, 1, :, :] = np.random.normal(0, 0.0001,
    #                                            sn_inc_conn[:, 1, :, :].shape)
    age2idxs['healthy'] = age2idxs[1] + age2idxs[2]
    idxs = age2idxs[age]
    sn_inc_conn = sn_inc_conn[idxs]

    n_sn = sn_inc_activity.shape[0]
    sns = [str(i) for i in range(n_sn) for _ in range(3)]
    incs = [1, 2, 3] * n_sn
    new_nrows = sn_inc_conn.shape[0]*sn_inc_conn.shape[1]
    nrois = sn_inc_conn.shape[-1]
    sn_inc_conn = np.reshape(sn_inc_conn, (new_nrows, nrois, nrois))

    cols = []
    vals = []
    for i in range(nrois):
        for j in range(i):
            cols.append(f'r{i}_{j}')
            vals.append(sn_inc_conn[:, i, j])
    vals = np.array(vals).T
    df = pd.DataFrame(vals, columns=cols)
    df['sn'] = sns
    df['inc'] = incs
    # sn_inc_conn = sn_inc_conn[df['inc'] != 2]

    # df = df[df['inc'] != 2]



    t_mat = np.full((nrois, nrois), np.nan)
    # p_mat = np.full((nrois, nrois), np.nan)
    p_l = []
    idx2roi = []

    formula_gen = '{ROI} ~ inc + sn'
    nans_mat = np.full((nrois, nrois), np.nan)
    for i in tqdm(range(nrois), desc='Looping lm'):
        # if i < 2: continue
        for j in range(i):
            # if (i, j) != (20, 19): continue
            idx2roi.append((i, j))
            n_nans = np.sum(pd.isna(df[f'r{i}_{j}']))
            nans_mat[i, j] = n_nans

            df_vars = df[['sn', 'inc', f'r{i}_{j}']].dropna()
            if len(df_vars) < 1:
                t_mat[i, j] = t_mat[j, i] = np.nan
                p_l.append(np.nan)
                continue

            formula = formula_gen.format(ROI=f'r{i}_{j}')
            model = smf.ols(formula=formula, data=df_vars)
            res = model.fit()
            # print(res.summary())
            # quit()
            p = res.pvalues.loc['inc']
            t = -res.tvalues.loc['inc']

            t_mat[i, j] = t_mat[j, i] = t
            # p_mat[i, j] = p_mat[j, i] = p
            p_l.append(p)
    #
    # plot_connectivity(nans_mat,
    #                   atlas['ticks'],
    #                   atlas['tick_labels'],
    #                   atlas['tick_lows'],
    #                   no_avg=True,
    #                   title=f'NaNs',
    #                   vmin=0, vmax=50,
    #                   cbar_label='z-score')
    # quit()

    p_mat = stats.t.cdf(t_mat, df=len(df)-1)
    z_mat = stats.norm.ppf(p_mat)
    # z_mat[t_mat > 0] *= -1

    p_l = np.array(p_l)
    return z_mat, t_mat, p_l, idx2roi

def get_paired_ttest_zs(fp='obj7_fMRI', combine_regions=False,
                        age='healthy'):
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'combine_regions': combine_regions
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    age2idxs['healthy'] = age2idxs[1] + age2idxs[2]
    idxs = age2idxs[age]
    M_graph, SD_graph, SE_graph, N_graph, t_graph, p_graph, z_graph = \
        get_stats_graphs(sn_inc_conn[idxs, 0, :, :],
                         sn_inc_conn[idxs, 1, :, :])

    trils = np.tril_indices_from(p_graph, k=-1)
    p_l = p_graph[trils]
    return z_graph, t_graph, p_l, trils



def plot_reg_zs(combine_regions=True, lm=True):
    if lm:
        z_mat, t_mat, p_l, idx2roi = \
            pickle_wrap(None, get_con_reg_zs,
                        kwargs={'combine_regions': combine_regions,
                                'age': 'healthy'},
                        easy_override=True)
    else:
        z_mat, t_mat, p_l, idx2roi = \
            pickle_wrap(None, get_paired_ttest_zs,
                        kwargs={'combine_regions': combine_regions,
                                'age': 'healthy'},
                        easy_override=False)

    p_l = p_l[~np.isnan(p_l)] / 2
    n_tests = np.sum(~np.isnan(p_l))
    # print(f'{n_tests=}')
    p_fwe = .05 / n_tests
    z_fwe = stats.norm.ppf(p_fwe)
    z_l = stats.norm.ppf(p_l)
    corr = 1 / (n_tests * p_l)
    for z, corr in zip(np.sort(z_l)[:10], np.sort(corr)[::-1]):
        print(f'{z=:.2f}, corr={corr:.2f}')

    sigs, p_corr, alpha_sidak, alpha_bon = \
        multipletests(p_l, alpha=.05, method='fdr_bh')
    if sum(sigs) == 0:
        cutoff_str = f'(NS, FWE_z = {z_fwe:.2f})'
    else:
        cutoff_z = round(np.min(np.abs(z_l[sigs])), 2)
        cutoff_str = f'(FDR, z > {cutoff_z})'

    print(f'{cutoff_str=}')

    atlas = get_atlas(combine_regions=combine_regions)
    atlas['ticks'] = atlas['ticks'][::2]
    atlas['tick_labels'] = atlas['tick_labels'][::2]
    atlas['tick_lows'] = atlas['tick_lows'][::2]

    plot_connectivity(z_mat,
                      atlas['ticks'],
                      atlas['tick_labels'],
                      atlas['tick_lows'],
                      no_avg=True,
                      title=f'Connectivity ~ congruency '
                            f'{cutoff_str}',
                      # vmin=-3, vmax=3,
                      cbar_label='z-score')

    # plot_connectivity(t_mat,
    #                   atlas['ticks'],
    #                   atlas['tick_labels'],
    #                   atlas['tick_lows'],
    #                   no_avg=True,
    #                   title=f't-val'
    #                         f'{cutoff_str}',
    #                   vmin=-3, vmax=3,
    #                   cbar_label='z-score')

# def t2z(t, df=999):
#     p = stats.t.cdf(t, df=df)
#     print(f'{p=:.3f}')
#     z = stats.norm.ppf(p)
#     print(f'{z=:.3f}')

if __name__ == '__main__':
    # t2z(-2.85)
    plot_reg_zs()
    # get_con_reg_zs()