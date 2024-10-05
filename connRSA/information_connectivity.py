from collections import defaultdict
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import jit, prange
from scipy import spatial
from scipy import stats
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.conn_regress import get_ERS_scores
from connRSA.conn_utils import get_BNA_ROIs
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan, get_IRAFs
from networks.old.networks import prep_networks

from old.plot_gen import plot_connectivity
from org_sns import get_sns
from organize_bhv import get_trial_info, sort_df_sn
from stim import get_semantic_vectors
from utils import pickle_wrap, stdize
from functools import cache
from sklearn import decomposition
import statsmodels.formula.api as smf
import seaborn as sns


# suppress RuntimeWarning: All-NaN slice
from warnings import filterwarnings
filterwarnings("ignore", category=RuntimeWarning,
               message="All-NaN slice encountered")


import os
os.chdir(r'/')

def rearrange_RSM():
    pass

@cache
def get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run, second_order,
                flat=True, within_nan=True, obj_sort=True):
    dir_in = fr'cache/conn_RSA/ars/RSA'
    RDM_method_ = 'within_nan'
    dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
    try:
        with open(fp_focus1, 'rb') as f:
            RSM = np.load(f, )
    except FileNotFoundError:
        # print('Not found!')
        RSM = np.full((114, 114), np.nan)
    except ValueError as e:
        print(f'{fp_focus1=}')
        print(f'allow_pickl=False ({sn}, {ROI}, {fp}): {e=}')
        quit()

    if within_nan:
        RSM = within_run_to_nan(RSM)

    df_sn = get_trial_info(sn, verbose=-1)
    df_sn, sess = sort_df_sn(df_sn, fp)

    if obj_sort:
        obj_idx = df_sn['obj'].argsort()
        RSM = RSM[np.ix_(obj_idx, obj_idx)]

    if flat:
        trils = np.tril_indices(RSM.shape[0], k=-1)
        return RSM[trils]
    else:
        return RSM

def get_IC_mat(sn, ROIs, fp, trial_similarity, stdize_by_run, second_order,
               within_nan=True):
    RSMs = []
    for ROI in ROIs:
        RSM = get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run,
                          second_order, within_nan=within_nan)

        RSMs.append(RSM)
    RSMs = np.array(RSMs)
    nan_cols = np.all(np.isnan(RSMs), axis=0)
    RSMs = RSMs[:, ~nan_cols]
    corr = np.ma.corrcoef(RSMs)
    corr[np.diag_indices_from(corr)] = np.nan
    return corr

def load_fp2RSMs(sn, ROIs, fps, trial_similarity, stdize_by_run,
                     second_order, within_nan=True, obj_sort=True):
    fp2RSMs = {}
    for fp in fps:
        RSMs = []
        for ROI in ROIs:
            RSM = get_ROI_RSM(sn, ROI, fp, trial_similarity, stdize_by_run,
                              second_order, within_nan=within_nan,
                              obj_sort=obj_sort)
            RSMs.append(RSM)
        RSMs = np.array(RSMs)
        fp2RSMs[fp] = RSMs
    return fp2RSMs

def get_cross_IC_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
                     second_order, within_nan=True, obj_sort=True,
                     same_RSM_corr=True):
    kw = {'ROIs': ROIs, 'trial_similarity': trial_similarity,
          'stdize_by_run': stdize_by_run, 'second_order': second_order,
          'within_nan': within_nan, 'sn': sn, 'fps': fps,
          'obj_sort': obj_sort}
    fp2RSMs = pickle_wrap(load_fp2RSMs, None, kwargs=kw,
                          easy_override=True, verbose=-1)

    t_st = time()
    l = []
    for fp0, RSMs0 in fp2RSMs.items():
        for fp1, RSMs1 in fp2RSMs.items():
            if fp0 == fp1:
                continue
            pair = [RSMs0, RSMs1]
            l.append(pair)
    l = np.array(l)
    # print(f'{same_RSM_corr=}')
    if same_RSM_corr:
        # Below code would make it so every RSM pair is calculated on the exact same
        # print(f'{l.shape=}')
        bad_cols = np.all(np.isnan(l), axis=(2)) # RSM cell is NaN for all ROI
        # print(f'{np.sum(bad_cols)=}')
        bad_cols = np.any(bad_cols, axis=(0, 1)) # any ROI has NaNs. Not great.
    else:
        bad_cols = np.all(np.isnan(l), axis=(0, 1, 2)) # RSM cell is NaN for all
        # Barely does anything if obj_sort = True. Just drops like 200
        #   drops like 2k if obj_sort = False
    print(f'{same_RSM_corr}, {np.sum(bad_cols)=}')

    l = l[:, :, :, ~bad_cols]
    l = stdize(l, axis=-1, nans=obj_sort)

    nan_ar = np.any(np.isnan(l), axis=(1))  # RSM cell is NaN for all ROI
    nan_ar = np.all(nan_ar, axis=1)

    corrs = numba_fp_x_fp_RSMs(l, nans=obj_sort, nan_ar=nan_ar)
    # print(corrs)
    t_end = time()
    print(f'Numba corr calc: {t_end - t_st:.3f}')
    return corrs

def get_cross_IRAF_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
                       second_order, RDM_method, semantic,
                       just_get_IRAFs=False):
    dir_root = fr'cache/conn_RSA/ars/RSA'
    # ROI2fps2iERS = {}
    num_pairs = (len(fps) * (len(fps) - 1)) // 2
    ROI2fps2iERS = np.full((num_pairs, len(ROIs), 114), np.nan)

    M_ERS_difs = []

    fp2stim_RSM = {}

    for fp0 in fps:
        dir0_focus = (f'{dir_root}/{fp0}_{trial_similarity}_'
                      f'{second_order}_{RDM_method}_{stdize_by_run}')
        fp_stim = f'{dir0_focus}/{sn}_stim_{semantic}.npy'
        with open(fp_stim, 'rb') as f:
            RSM_stim = np.load(f)
        if RDM_method == 'within_nan':
            RSM_stim = within_run_to_nan(RSM_stim)
        fp2stim_RSM[fp0] = RSM_stim

    df_sn = get_trial_info(sn)

    ROI2fp2IRAFs = defaultdict(dict)

    # ar = np.empty((246, 4, 114))
    ar = np.full((246, len(fps), 114), np.nan)

    for i, ROI in enumerate(ROIs):
        # ROI2fps2iERS[ROI] = {}
        # l_ERS_dif_M = []
        cnt = 0
        for j, fp0 in enumerate(fps):
            dir0_focus = (f'{dir_root}/{fp0}_{trial_similarity}_'
                          f'{second_order}_{RDM_method}_{stdize_by_run}')
            fp0_focus = f'{dir0_focus}/{sn}_{ROI}_BOLD.npy'
            try:
                with open(fp0_focus, 'rb') as f:
                    RSM0_focus = np.load(f)
                if RDM_method == 'within_nan':
                    RSM0_focus = within_run_to_nan(RSM0_focus)
            except FileNotFoundError:
                # print(f'Missing: {ROI}, {fp0=}')
                ar[i, j, :] = np.full(114, np.nan)
                continue

            RSM_stim = fp2stim_RSM[fp0]
                # get_IRAFs sorts by df_sn['obj']
            IRAFs = get_IRAFs(RSM0_focus, RSM_stim, df_sn, within_to_nan=True,
                              by_run=False, second_order='spear')
            ROI2fp2IRAFs[ROI][fp0] = IRAFs
            ar[i, j, :] = IRAFs

    if get_IRAFs:
        ar = np.transpose(ar, (1, 0, 2))
        return ar

    ar = stdize(ar, axis=-1)
    t_st = time()
    mats = numba_IRAF_x_IRAF(ar)
    t_end = time()
    print(f'Numba IRAF x IRAF calc: {t_end - t_st:.3f} s')
    return mats

# @jit(nopython=True, parallel=True, fastmath=True)
def numba_IRAF_x_IRAF(ar):
    # ar.shape = (246, 4, 114)
    num_ROIs = ar.shape[0]
    num_fps = ar.shape[1]
    # num_trials = ar.shape[2]
    num_prods = num_fps * (num_fps - 1) # // 2

    if num_fps == 4:
        lookup = {(1, 0): 0, (2, 0): 1, (2, 1): 2,
                  (3, 0): 3, (3, 1): 4, (3, 2): 5,

                  (0, 1): 6, (0, 2): 7, (1, 2): 8,
                  (0, 3): 9, (1, 3): 10, (2, 3): 11
                  }
    elif num_fps == 3:
        lookup = {(1, 0): 0, (2, 0): 1, (2, 1): 2,
                  (0, 1): 3, (0, 2): 4, (1, 2): 5,}
    else:
         raise ValueError

    assert num_fps == 4 or num_fps == 3
    # ar_out = np.full((num_prods, num_ROIs, num_ROIs), np.nan)
    ar_out = np.empty((num_prods, num_ROIs, num_ROIs)
                      ) # 0.2s faster than np.full(..., nan). i live for danger.

    for ROI_i in prange(num_ROIs):
        for ROI_j in range(num_ROIs):
            for fp0 in range(num_fps):
                IRAFs0 = ar[ROI_i, fp0, :]
                for fp1 in range(fp0):
                    IRAFs1 = ar[ROI_j, fp1, :]
                    r = np.mean(IRAFs0 * IRAFs1)
                    ar_out[lookup[(fp0, fp1)], ROI_i, ROI_j,] = r
                    ar_out[lookup[(fp1, fp0)], ROI_j, ROI_i,] = r
                    # ar_out[lookup[(fp0, fp1)], ROI_j, ROI_i, ] = r
    return ar_out


def get_cross_ERS_mat(sn, ROIs, fps, trial_similarity, stdize_by_run,
                      cross=False, nan_block=True, do_same=False,
                      get_var=False):
    dir_root = fr'cache/conn_RSA/ars/ERS'
    # ROI2fps2iERS = {}
    num_pairs = (len(fps) * (len(fps) - 1)) // 2
    fp2ROI2iNPS = np.full((num_pairs, len(ROIs), 114), np.nan)


    tup2cnt = {}
    fp12block2objs = {}

    M_ERS_difs = []
    for i, ROI in enumerate(ROIs):
        # ROI2fps2iERS[ROI] = {}
        # l_ERS_dif_M = []
        cnt = 0
        for fp0 in fps:
            for fp1 in fps:
                if fp0 >= fp1: continue
                dir_focus = (fr'{dir_root}/{fp0}_{fp1}_{trial_similarity}_'
                             fr'{stdize_by_run}')
                fp = f'{dir_focus}/{sn}_{ROI}_BOLD.npy'

                if nan_block:
                    if fp1 not in fp12block2objs:
                        df_sn = get_trial_info(sn, verbose=-1)
                        df_sn.sort_values(by='obj', inplace=True)
                        sess1 = (fp1.split('_')[0].replace('2', '').
                                 replace('3', '').replace('4', '').
                                 replace('7', '').replace('8', ''))
                        df_sn[f'{sess1}_trial'] = (
                            stats.rankdata(df_sn[f'{sess1}_trial']))
                        df_sn[f'{sess1}_trial'] = (df_sn[f'{sess1}_trial'].
                                                   astype(int))
                        obj2block_l = []
                        block2objs = defaultdict(list)
                        for j in range(len(df_sn)):
                            row = df_sn.iloc[j]
                            block = (row[f'{sess1}_trial'] - 1) // 38
                            obj2block_l.append(block)
                            block2objs[block].append(j)
                        block2objs[0] = set(block2objs[0])
                        block2objs[1] = set(block2objs[1])
                        block2objs[2] = set(block2objs[2])
                        fp12block2objs[fp1] = block2objs
                try:
                    with open(fp, 'rb') as f:
                        ERS = np.load(f)

                    if nan_block:
                        for j in range(len(obj2block_l)):
                            for k in range(114):
                                if (k in block2objs[obj2block_l[j]]):
                                    continue

                                ERS[k, j] = np.nan

                    ERS_dif, _ = get_ERS_scores(ERS, get_same=do_same)

                except EOFError:
                    ERS_dif = np.full(114, np.nan)

                except FileNotFoundError:
                    ERS_dif = np.full(114, np.nan)

                fp2ROI2iNPS[cnt, i, :] = ERS_dif
                tup = tuple(sorted((fp0, fp1)))
                tup2cnt[tup] = cnt
                cnt += 1
        if np.any(np.isnan(fp2ROI2iNPS[:, i, :])):
            M_ERS_difs.append(np.nanmean(fp2ROI2iNPS[:, i, :]))
        else:
            M_ERS_difs.append(np.mean(fp2ROI2iNPS[:, i, :]))

    if cross:
        valid_pairs = set()
        for fp0 in fps:
            for fp1 in fps:
                if fp0 == fp1: continue
                tup01 = (fp0, fp1)
                tup01 = tuple(sorted(tup01))
                for fp2 in fps:
                    for fp3 in fps:
                        if fp2 == fp3: continue
                        tup23 = (fp2, fp3)
                        tup23 = tuple(sorted(tup23))
                        if set(tup01).intersection(tup23):
                            continue
                        valid_pairs.add((tup01, tup23))
        valid_pairs = sorted(list(valid_pairs))
        # print(valid_pairs)
        # quit()
        pair2fp2ROI2iNPS = np.full((len(valid_pairs), 2, len(ROIs), 114), np.nan)
        for i, (tup01, tup23) in enumerate(valid_pairs):
            cnt = tup2cnt[tup01]
            cnt2 = tup2cnt[tup23]
            pair2fp2ROI2iNPS[i, 0, :, :] = fp2ROI2iNPS[cnt, :, :]
            pair2fp2ROI2iNPS[i, 1, :, :] = fp2ROI2iNPS[cnt2, :, :]
        pair2fp2ROI2iNPS = stdize(pair2fp2ROI2iNPS, axis=-1)
        corrs = numba_fpfp_x_fpfp_NPS(pair2fp2ROI2iNPS)
        # print(corrs.shape)
        # diag = np.diag(np.nanmean(corrs, axis=0))
        # print(f'{diag=}')
        # for i, corr in enumerate(corrs):
        #     plt.title(f'{i=}')
        #     plt.imshow(corr)
        #     plt.show()
        # plt.imshow(np.nanmean(corrs, axis=0))
        # plt.show()
        # quit()
    else:

        if get_var == 'all':
            return fp2ROI2iNPS
        elif get_var != False:
            idxs = get_idxs(get_var)
            return fp2ROI2iNPS[:, idxs, :]
            # v = np.nanvar(fp2ROI2iNPS[:, idxs, :], axis=1)
            # return v
            # return fp2ROI2iNPS
        fp2ROI2iNPS = stdize(fp2ROI2iNPS, axis=-1)

        # print(fp2ROI2iNPS.shape)

        # print(f'{fp2ROI2iNPS.shape=}')
        # quit()
        corrs = numba_fp_x_fp_ERS(fp2ROI2iNPS)

    return corrs, M_ERS_difs

@jit(nopython=True, parallel=True, fastmath=True)
def numba_fpfp_x_fpfp_NPS(l):
    num_fpfp_pairs = l.shape[0]
    num_ROIs = l.shape[2]
    out = np.empty((num_fpfp_pairs * 2, num_ROIs, num_ROIs))
    for i in prange(num_fpfp_pairs):
        for j0 in range(num_ROIs):
            ERS0 = l[i, 0, j0, :]
            for j1 in range(num_ROIs):
                out[i, j0, j1] = np.mean(ERS0 * l[i, 1, j1, :])

            ERS0 = l[i, 1, j0, :]
            for j1 in range(num_ROIs):
                out[i + 6, j0, j1] = np.mean(ERS0 * l[i, 0, j1, :])
    return out

@jit(nopython=True, parallel=True, fastmath=True)
def numba_fp_x_fp_ERS(l):
    num_fp_pairs = l.shape[0]
    num_ROIs = l.shape[1]
    out = np.empty((num_fp_pairs, num_ROIs, num_ROIs))
    for i in prange(num_fp_pairs):
        for j0 in range(num_ROIs):
            ERS0 = l[i, j0, :]
            for j1 in range(num_ROIs):
                out[i, j0, j1] = np.mean(ERS0 * l[i, j1, :])
    return out

@jit(nopython=False, parallel=True, fastmath=True)
def numba_fp_x_fp_RSMs(l, nans=False, nan_ar=True): # 4 seconds first then 3 seconds vs. 9 w/ numpy below
    num_fp_pairs = l.shape[0]
    num_ROIs = l.shape[2]
    out = np.empty((num_fp_pairs, num_ROIs, num_ROIs))
    for i in prange(num_fp_pairs): # prange gives an 8x speedup??
        RSM0 = l[i][0]
        RSM1 = l[i][1]
        for j0 in range(num_ROIs):
            for j1 in range(num_ROIs):
                if nans:
                    either_nan = nan_ar[i, :]
                    out[i, j0, j1] = np.mean(RSM0[j0, ~either_nan] *
                                             RSM1[j1, ~either_nan])
                else:
                    out[i, j0, j1] = np.mean(RSM0[j0, :] * RSM1[j1, :])
    return out

def numpy_fp_x_fp_RSMs(l): # if numba doesn't use prange, this is same speed
    num_fp_pairs = l.shape[0]
    num_ROIs = l.shape[2]
    out = np.empty((num_fp_pairs, num_ROIs, num_ROIs))
    for i in range(num_fp_pairs):
        RSM0 = l[i][0]
        RSM1 = l[i][1]
        corr = np.mean(RSM0[None, :, :] * RSM1[:, None, :], axis=-1)
        out[i] = corr
    return out

# calculate IC between tasks and normalize by an ROIs IC with itself
#   this is needed to rule out that high/low IC is due to data reliablity
#   e.g., Occipital 1 may have higher IC with occipital 2 than IPL 1 has with
#   IPL 2, but this could be due to occipital 1 & 2 having more reliable data

def get_idxs(ROI):
    regions = set(prep_networks(
        network_setting=ROI2NETWORK[ROI])[ROI])
    ROIs_match = [i for region in regions
                  for i, ROI in enumerate(get_BNA_ROIs()) if region in ROI]
    return ROIs_match


def plot_network_M(corrs, corrs_FC, sns_FC, fn_out):

    # ventral_idxs = get_idxs('ITL')
    # occ_idxs = get_idxs('Occipital')
    # corrs_FC[:, *np.diag_indices(corrs_FC.shape[1])] = np.nan
    # corrs_ventral = corrs_FC[:, ventral_idxs][:, :, ventral_idxs]
    # M_ventral_FC = np.nanmean(corrs_ventral, axis=(1, 2))
    # corrs_occ = corrs_FC[:, occ_idxs][:, :, occ_idxs]
    # M_occ_FC = np.nanmean(corrs_occ, axis=(1, 2))
    #
    # print(f'Ventral: {np.nanmean(M_ventral_FC):.3f} '
    #       f'({np.nanstd(M_ventral_FC):.3f})')
    # print(f'Occ: {np.nanmean(M_occ_FC):.3f} ({np.nanstd(M_occ_FC):.3f})')
    # t, p = stats.ttest_rel(M_ventral_FC, M_occ_FC)
    # N = M_ventral_FC.shape[0]
    # d = t / np.sqrt(N)
    # print(f'Ventral vs. occ ({N=}): {t=:.3f}, {p=:.3f}, {d=:.3f}')
    # print(f'{len(sns_FC)=}')
    # M_ventral_FC = [M_ventral_FC[sns_FC.index(sn)] for sn in sns]
    # M_occ_FC = [M_occ_FC[sns_FC.index(sn)] for sn in sns]
    # t_FC, p_FC = stats.ttest_rel(M_ventral_FC, M_occ_FC)
    # print(f'vetral vs. occ FC: {t_FC=:.3f}, {p_FC=:.3f}')

    networks = ['Occipital', 'ITL', 'Parietal', 'PFC']
    network2name = {'Occipital': 'Occipital', 'ITL': 'Temporal',
                    'Parietal': 'Parietal', 'PFC': 'PFC'}
    names = [network2name[net] for net in networks]
    net2idxs = {}
    df_as_l = defaultdict(list)

    colors = ['dodgerblue', 'darkorange', 'crimson', 'limegreen']
    # plt.gcf().add_axes([0.1,0.1, 0.35,0.8])

    for net in networks:
        idxs = get_idxs(net)
        net2idxs[net] = idxs
        net_corr = corrs[:, idxs][:, :, idxs]
        net_sn_vals = np.nanmean(net_corr, axis=(1, 2))
        net_M = np.nanmean(net_sn_vals)
        net_SD = np.nanstd(net_sn_vals, ddof=1)
        net_SE = net_SD / np.sqrt(np.sum(~np.isnan(net_sn_vals)))
        df_as_l['net'].extend([network2name[net]]*len(sns_FC))
        df_as_l['sn'].extend(sns_FC)
        df_as_l['val'].extend(net_sn_vals)

    df = pd.DataFrame(df_as_l)
    plt.figure(figsize=(3.5, 4))
    plt.rcParams.update({'font.size': 14})
    bp = sns.stripplot(x='net', y='val', #hue='net', #palette=colors,
                       alpha=0.5, linewidth=.7,
                       data=df, order=names,
                       palette=colors)
    plt.ylabel('Mean correlation', fontsize=14)
    plt.xlabel('')
    # plt.xticks([0, 1, 2, 3], networks, color=colors)
    plt.xticks([0, 1, 2, 3], ['', '', '', ''])
    plt.xlim(-0.5, 3.65)
    plt.tick_params(bottom=False)
    # plt.gca().spines[['top', 'right']].set_visible(False)
    plt.gca().spines[['top', 'left', 'bottom']].set_visible(False)
    plt.gca().yaxis.set_label_position('right')
    plt.gca().set_ylabel('Mean correlation', rotation=270, labelpad=20)
    plt.gca().yaxis.tick_right()

    colors = ['dodgerblue', 'darkorange', 'crimson', 'green']
    for i, name in enumerate(names):
        # plt.gca().get_xticklabels()[i].set_color(colors[i])
        df_net = df[df['net'] == name]
        # lowest_net = df_net['val'].min()
        highest_net = df_net['val'].max()
        # plt.text(i, lowest_net - 0.025, f'{name}', ha='center',
        #          color=colors[i])
        plt.text(i, highest_net + 0.0175, f'{name}', ha='center',
                 fontsize=14, color=colors[i])

    df_OC = df[df['net'] == 'Occipital']['val'].values
    df_ITL = df[df['net'] == 'Temporal']['val'].values
    t, p = stats.ttest_rel(df_OC, df_ITL)
    d = t / np.sqrt(len(df_OC))
    print(f'{fn_out}: {d=:.3f}')


    # plt.text(0)
    # plt.title(title)
    if 'regressed' in fn_out:
        pass
    else:
        if 'NPS' in fn_out:
            plt.ylim(0.0, 0.345)
            plt.gca().set_yticks([0.0, 0.1, 0.2, 0.3])
        elif 'RSM' in fn_out:
            plt.ylim(0, None)
        else:
            plt.ylim(0, None)
            # plt.ylim(0.02, 0.62)

    plt.yticks(fontsize=14)

    # plt.axis([-0.5, 3.5, 0, 0.25])
    # plt.axis('equal')
    # plt.axis([-0.5, 3.5, 0, 0.25], frameon=False)
    # plt.show()

    # ax2 = plt.gcf().add_axes([0.2, 0.2, 0.1, 0.8])
    # ax2 = plt.gcf().add_axes([0.1, 0.1, 0.01, 0.8])
    # ax2.set_ylabel('Mean correlation')
    # ax2.set_ylim(0.05, 0.35)

    # ax2.spines[['top', 'right', 'bottom']].set_visible(False)
    # ax2.set_ylim(0, 0.25)
    plt.tight_layout()
    fp_fig = fr'connRSA/FC_figs/M_{fn_out}'
    plt.savefig(fp_fig, dpi=300)
    plt.show()

    # quit()

        # print(f'{net}: {net_M=:.3f} [{net_SE:.3f}]')
    # quit()

    # ventral_idxs = get_idxs('ITL')
    # occ_idxs = get_idxs('Occipital')
    # corrs_ventral = corrs[:, ventral_idxs][:, :, ventral_idxs]
    # M_ventral = np.nanmean(corrs_ventral, axis=(1, 2))
    # corrs_occ = corrs[:, occ_idxs][:, :, occ_idxs]
    # M_occ = np.nanmean(corrs_occ, axis=(1, 2))

    # print(f'Ventral: {np.nanmean(M_ventral):.3f} ({np.nanstd(M_ventral):.3f})')
    # print(f'Occ: {np.nanmean(M_occ):.3f} ({np.nanstd(M_occ):.3f})')
    # t, p = stats.ttest_rel(M_ventral, M_occ)
    # N = M_ventral.shape[0]
    # d = t / np.sqrt(N)
    # print(f'Ventral vs. occ ({N=}): {t=:.3f}, {p=:.3f}, {d=:.3f}')


    # IC = np.concatenate([M_ventral, M_occ])
    # FC = M_ventral_FC + M_occ_FC
    # try:
    #     df = pd.DataFrame({'IC': IC, 'FC': FC,
    #                        'location': ['Ventral'] * len(M_ventral) +
    #                                    ['Occipital'] * len(M_occ),
    #                        'sn': sns_FC + sns_FC})
    #     formula = r'IC ~ sn + location + FC' # FC + FC +
    #     mod = smf.ols(formula=formula, data=df)
    #     res = mod.fit()
    #     print(res.summary())
    # except ValueError as e:
    #     print(f'{e=}')


def run_IC_analysis(ERS=True, regress_FC=True):
    # semantic = False
    # regress_FC = False

    dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)


    # ERS = False
    ERS_nan_block = False

    cross = False
    IRAF = False

    semantic = False
    drop_con = False
    same_RSM_corr = False
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    second_order = 'spear'
    atlas = get_atlas()
    ROIs = atlas['ROIs']
    # print(f'{ROIs=}')
    # quit()
    if cross and ERS:
        assert not drop_con

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117',
           '118', '119', '120', '123', '124', '126', '127', '128', '129',
           '130', '131', '132', '134', '135', '136',
           '137', '138', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214',
           '216', '217', '218', '219', '221', '222', '224', '225', '227',
           '230', '232', '233', '234', '235', '239']

    four_tasks = '7'
    fps = prep_fps(four_tasks)

    kwargs = {'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'ROIs': ROIs,
              }

    corrs = []
    # TODO: maybe regress out the activation normal FC matrix?

    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]
    ERS_scores_all = []

    for i, sn in tqdm(enumerate(sns), desc=f'Looping IC: {cross=}'):

        kwargs['sn'] = sn
        if IRAF:
            kwargs['fps'] = fps
            kwargs['second_order'] = 'spear'
            kwargs['RDM_method'] = 'within_nan'
            kwargs['semantic'] = semantic
            sn_corrs = pickle_wrap(get_cross_IRAF_mat,
                                   kwargs=kwargs, verbose=-1,
                                   easy_override=False,
                                   dt_max=dt_max)

        elif ERS:
            kwargs['fps'] = fps
            kwargs['cross'] = cross
            kwargs['nan_block'] = ERS_nan_block

            sn_corrs, ERS_scores = pickle_wrap(get_cross_ERS_mat,
                                               kwargs=kwargs, verbose=-1,
                                               easy_override=False,
                                               dt_max=dt_max)
            ERS_scores_all.append(ERS_scores)
        elif cross:
            kwargs['second_order'] = second_order
            kwargs['within_nan'] = True
            kwargs['fps'] = fps
            kwargs['same_RSM_corr'] = same_RSM_corr
            sn_corrs = pickle_wrap(get_cross_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False, dt_max=dt_max)
        else:
            kwargs['second_order'] = second_order
            kwargs['within_nan'] = True
            sn_corrs = []
            for fp in fps:
                kwargs['fp'] = fp
                corr = pickle_wrap(get_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False, dt_max=dt_max)
                sn_corrs.append(corr)
            sn_corrs = np.array(sn_corrs)

        corr = np.nanmean(sn_corrs, axis=0)

        corrs.append(corr)


    corrs = np.array(corrs)
    corrs[:, *np.diag_indices(corrs.shape[1])] = np.nan

    corrs_FC, M_ventral_FC, M_occ_FC, sns_FC = (
        pickle_wrap(plot_FC_mat, None, easy_override=False, dt_max=dt_max))

    title = 'RSM x RSM connectivity' if not ERS else 'NPS x NPS connectivity'
    if regress_FC:
        title += ' (FC regressed)'
    fn_out = r'NPS' if ERS else 'RSM'
    fn_out += '_regressed' if regress_FC else ''
    fn_out += '.png'

    atlas = get_atlas(lifu_labels=False)

    plot_network_M(corrs, corrs_FC, sns_FC, fn_out)

    if regress_FC:
        for sn_i in range(corrs.shape[0]):
            corr = corrs[sn_i]
            corr_flat = corr.flatten()
            corr_FC = corrs_FC[sn_i]
            corr_FC_flat = corr_FC.flatten()
            nans = np.isnan(corr_flat) | np.isnan(corr_FC_flat)
            corr_flat_ = corr_flat[~nans]
            corr_FC_flat_ = corr_FC_flat[~nans]
            slope, intercept, r, p, se = (
                stats.linregress(corr_FC_flat_, y=corr_flat_,
                                 alternative='two-sided'))

            corr_flat -= slope * corr_FC_flat
            corr = corr_flat.reshape(corr.shape)
            corrs[sn_i] = corr



    M = np.nanmean(corrs, axis=0)
    SE = stats.sem(corrs, axis=0, nan_policy='omit')
    N = np.sum(~np.isnan(corrs), axis=0)

    for i in range(N.shape[-1]):
        val = 60 - N[i, 0]
        if i == 0 or val == 0:
            continue
        print(f'{i}, {ROIs[i]}: {val}')

    M[N <= 48] = np.nan

    fn_out = r'NPS' if ERS else 'RSM'
    fn_out += '_regressed' if regress_FC else ''
    fn_out += '.png'
    fp_fig = fr'connRSA/FC_figs/{fn_out}'

    plot_connectivity(M, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'], title=title, tile=.01,
                      no_avg=True, cbar_label='Correlation (r)', fp=fp_fig)


def plot_FC_mat(drop_con=False, four_tasks='7'):
    fps = prep_fps(four_tasks)
    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]

    corrs = []
    sns = None
    prev_sns = None
    # fps = ['obj7_fMRI'] # may have caused error in figure being true
    for fp in fps:
        kwargs = {'fp': fp,
                  'split': False,
                  'key': 'inc',
                  'key_vals': (1, 2, 3),
                  'strict_sns': True,
                  'get_df_sn': True
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
            pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                        easy_override=False, verbose=1, cache_dir='cache',
                        RAM_cache=True)
        sns = [df_sn['sn'].iloc[0] for df_sn in df_sns]
        if prev_sns is None:
            prev_sns = sns
        else:
            assert tuple(prev_sns) == tuple(sns)
        corrs.append(sn_conn)

    corrs = np.nanmean(corrs, axis=0)
    M = np.nanmean(corrs, axis=0)
    atlas = get_atlas(lifu_labels=False)
    fn_out = r'basic_FC.png'
    fp_fig = fr'connRSA/FC_figs/{fn_out}'
    plot_network_M(corrs, None, sns, fn_out)

    plot_connectivity(M, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title='Functional connectivity', fp=fp_fig,
                      no_avg=True, cbar_label='Correlation (r)',)


    return corrs, sns


if __name__ == '__main__':
    # plot_FC_mat(drop_con=False, four_tasks='7')
    # quit()

    plt.rcParams.update({'font.sans-serif': 'Arial'})

    run_IC_analysis(ERS=True, regress_FC=True)
    run_IC_analysis(ERS=False, regress_FC=True)
    # plot_FC_mat()
