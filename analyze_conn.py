import random

from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np

from DNN_vectors import get_DNN_vecs, get_img_fns
from basic_fCon import get_FC
from fMRI_analysis import get_ROI_info, plot_connectivity, regress_out, stdize
from organize_bhv import get_trial_info
from nilearn import image, datasets
from glob import glob

from wordvec_get_vectors import get_semantic_vectors
import matplotlib.pyplot as plt
import utils
import scipy.stats as stats

from scipy import io
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pickle
import matplotlib
from statsmodels.stats.multitest import multipletests

def plot_test():
    age = 1
    cin = None
    semantic = False
    early_late = True
    fp_out = get_cache_RSA_fp(cin, age, semantic, early_late)
    with open(fp_out, 'rb') as file:
        d = pickle.load(file)

    d_IRAF_conn = d['IRAF_conn']
    mat0 = d_IRAF_conn['obj']
    mat1 = d_IRAF_conn['scn']
    mat2 = d_IRAF_conn['dif']
    mat3 = d_IRAF_conn['dif_']

    fig, axs = plt.subplots(1, 4, figsize=(24, 7))
    ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = get_ROI_info()
    plot_connectivity(mat0, ticks, tick_labels, tick_lows, title='obj',
                      ax=axs[0])
    plot_connectivity(mat1, ticks, tick_labels, tick_lows, title='scene',
                      ax=axs[1])
    plot_connectivity(mat2, ticks, tick_labels, tick_lows, title='dif',
                      ax=axs[2])
    plot_connectivity(mat3, ticks, tick_labels, tick_lows, title='dif reg',
                      ax=axs[3])
    plt.tight_layout()
    plt.show()

def get_age_str(age):
    return 'healthy' if age == 'healthy' else 'YA' if age == 1 else 'OA'

def get_cin_str(cin):
    return '' if cin is None else \
        '_Con' if cin == 1 else \
        '_Inc' if cin == 2 else \
        '_Neu' if cin == 3 else 'BAD_CIN'

def get_cache_RSA_fp(cin, age, semantic, early,
                     pre_str='', rxr=False):
    age_str = 'healthy' if age == 'healthy' else \
        'YA' if age == 1 else 'OA'
    cin_str = '' if cin is None else \
        '_Con' if cin == 1 else \
        '_Inc' if cin == 2 else \
        '_Neu' if cin == 3 else 'BAD_CIN'
    sem_str = '_sem' if semantic else ''
    el_str = '' if semantic else '_early' if early else '_late'
    rxr_str = '_rxr' if rxr else ''
    fp_out = fr'cache/RSA/{age_str}{cin_str}{sem_str}{el_str}{rxr_str}.pkl'
    return fp_out

def regress_out_normal_connectivity(mat, age, cin):
    mat = np.array(mat)
    age_str = get_age_str(age)
    cin_str = get_cin_str(cin)
    fp_FC = fr'cache/fCon_{age_str}{cin_str}.pkl'
    FC = pickle_wrap(fp_FC, lambda: get_FC(age, cin),
                     easy_override=False)
    n_ROIs = FC.shape[1]
    for i in range(n_ROIs):
        for j in range(n_ROIs):
            if i == j:
                continue
            print(mat[:, i, j])
            plt.scatter(FC[:, i, j], mat[:, i, j])
            plt.show()
            mat[:, i, j] = regress_out(FC[:, i, j], mat[:, i, j])
            print(mat[:, i, j])
            quit()
    return mat

def make_title_str(pre_str, key, age, early, semantic, cin):
    if key == 'obj':
        key_str = 'Object RSA'
    elif key == 'scn':
        key_str = 'Scene RSA'
    elif key == 'dif':
        key_str = 'Difference RSA'
    elif key == 'dif_':
        key_str = 'Difference RSA (regressed)'
    else:
        key_str = ''
    age_str = 'YA & OA' if age == 'healthy' else 'YA' if age == 1 else 'OA'
    if semantic:
        rsa_str = 'word2vec'
    else:
        rsa_str = '1st-layer DNN' if early else 'late-layer DNN'
    cin_str = 'Con, Inc, & Neu' if cin is None \
        else 'Con' if cin == 1 else 'Inc' if cin == 2 else 'Neu'
    out_str = f'{pre_str} {key_str}. {age_str}. {rsa_str}. {cin_str}'
    return out_str

# TODO: within-ROI voxel-voxel connectivity

def analyze_ROIs(age='healthy', early=True, semantic=False, cin=None):
    font = {'size': 14}
    matplotlib.rc('font', **font)
    cmap = plt.get_cmap('turbo')

    fp1 = get_cache_RSA_fp(cin=cin, age=age, semantic=semantic, early=early)
    with open(fp1, 'rb') as file:
        d1 = pickle.load(file)
    key0 = 'z'
    key1 = 'scn'
    ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = get_ROI_info()
    colors = cmap(np.linspace(0, 1, len(ticks)))
    idxs = list(np.arange(len(colors)))
    idxs_ = idxs.copy()

    idxs_[::2] = idxs[:len(idxs_)//2 + 1]
    idxs_[1::2] = idxs[len(idxs_)//2 + 1:]
    # random.shuffle(idxs)
    colors = colors[idxs_]
    colors[:, :3] /= 1.3
    # print(colors)
    # quit()
    region2color = dict(zip(tick_labels, colors))
    # print(region2color)
    # quit()
    # print(tick_labels)
    # quit()
    Ms = []
    colors = []
    ps = []
    for ROI in ROIs:
        ROI_num, ROI_str = ROI.split(' ')
        region = ROI_str.split('_')[0]
        color = region2color[region]
        colors.append(color)
        M0 = np.mean(d1[key0][key1][ROI])
        SD = np.std(d1[key0][key1][ROI])
        N = len(d1[key0][key1][ROI])
        SE = SD / np.sqrt(N)
        t = M0 / SE
        p = stats.t.sf(np.abs(t), N-1)*2
        ps.append(p)
        Ms.append(t)
    alpha = .10

    sigs, p_corr, alpha_sidak, alpha_bon = multipletests(ps, alpha=alpha,
                                                         method='fdr_bh')
    if np.min(p_corr) < alpha:
        p_corr_ = p_corr.copy()
        # print(p_corr)
        p_corr_[p_corr_ > alpha] = 0
        narrowest_cutoff = np.argmax(p_corr_)
        # print(narrowest_cutoff)
        # print(p_corr[narrowest_cutoff])
        t_cutoff = Ms[narrowest_cutoff]
        plt.plot([0, len(Ms)], [t_cutoff, t_cutoff], 'k--', linewidth=1)

    plt.scatter(ROI_nums, Ms, color=colors, s=10)
    min_val = np.min(Ms)
    max_val = np.max(Ms)
    plt.ylim([min_val*1.02, max_val*1.02])
    # for i in range(len(tick_lows)):
    #     plt.plot([tick_lows[i], tick_lows[i]], [min_val, max_val], 'k--',
    #              zorder=-10)
    plt.plot([0, len(Ms)], [0, 0], color='k', zorder=-1, linewidth=1)
    plt.xticks(ticks, tick_labels, rotation=90, fontsize=10)
    plt.ylabel('t-value')
    title_str = make_title_str('', key1, age, early,
                               semantic, cin)
    plt.title(title_str, fontsize=11.5)

    # plt.gca().tick_params(axis='x', colors=colors)

    for i in range(len(ticks)):
        # print(tick_labels[i])
        plt.gca().get_xticklabels()[i].set_color(region2color[tick_labels[i]])
    plt.show()
    quit()

def CIN_compare(age=1, early=False, semantic=False):
    fp1 = get_cache_RSA_fp(cin=1, age=age, semantic=semantic, early=early)
    fp2 = get_cache_RSA_fp(cin=2, age=age, semantic=semantic, early=early)
    with open(fp1, 'rb') as file:
        d1 = pickle.load(file)
    with open(fp2, 'rb') as file:
        d2 = pickle.load(file)

    # mat = regress_out_normal_connectivity(d1['IRAF_conn']['obj'], age, 1)

    key0 = 'z'
    key1 = 'dif_'
    ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = get_ROI_info()
    # plot_connectivity(mat, ticks, tick_labels, tick_lows, title='obj')
    # quit()

    # for key2 in ['obj', 'scn', 'dif', 'dif_']:
    for ROI in ROIs:
        M0 = np.mean(d1[key0][key1][ROI])
        M1 = np.mean(d2[key0][key1][ROI])
        l0 = np.array(d1[key0][key1][ROI])
        l1 = np.array(d2[key0][key1][ROI])
        t, p = stats.ttest_ind(l0, l1)
        print(f'{ROI}: {M0:.3f} {t=:.3f} ')

def replace_w_nan_if_needed(vals):
    clean = []
    for x in vals:
        if x.shape == vals[0].shape:
            clean.append(x)
        else:
            shape_nan = (vals[0].shape[0] - x.shape[0], vals[0].shape[1])
            fill_nan = np.full(shape_nan, np.nan)
            x = np.concatenate([x, fill_nan])
            clean.append(x)
            # print(f'{x.shape=}')
            # quit()
            # clean.append(np.full(vals[0].shape, np.nan))
    return np.array(clean)

def test_IRAF_x_activity(age='healthy', early=True, semantic=False):
    fp = get_cache_RSA_fp(cin=None, age=age, semantic=semantic, early=early)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = get_ROI_info()

    IRAFs = [np.array(d['IRAFs_ROI']['dif_'][roi0]) for roi0 in ROIs]
    IRAFs = replace_w_nan_if_needed(IRAFs)
    # print(f'{IRAFs.shape=}')
    # quit()
    # IRAFs = np.array(list(filter(lambda x: x.shape[0] == 33, IRAFs)))
    activity = [np.array(d['activity'][roi1]) for roi1 in ROIs]
    activity = replace_w_nan_if_needed(activity)
    # activity = np.array(list(filter(lambda x: x.shape[0] == 33, activity)))
    r_Ms, r_SDs, t = bulk_correlate(IRAFs, activity)
    # quit()
    #
    # mat = []
    # for roi0 in ROIs:
    #     IRAF = d['IRAFs_ROI']['obj'][roi0]
    #     print(f'{IRAF.shape=}')
    #     IRAF_m = np.mean(IRAF, axis=1)
    #     # print(f'{IRAF.shape=}')
    #     # print(f'{IRAF.shape=}')
    #     # quit()
    #     t_iraf = np.mean(IRAF_m, axis=0) / np.std(IRAF_m, axis=0) * np.sqrt(len(IRAF))
    #     if abs(t_iraf) > 2.0:
    #         print(f'IRAF {roi0} {t_iraf=:.3f}')
    #     # if IRAF.shape[0] != 56: continue
    #     # quit()
    #     v = []
    #     for roi1 in ROIs:
    #         # if roi0 != roi1: continue
    #         d['activity'][roi1] = np.array(d['activity'][roi1])
    #         activity = d['activity'][roi1]
    #         print(f'{activity.shape=}')
    #         if activity.shape[0] != len(IRAF):
    #             v.append(np.nan)
    #             continue
    #         # quit()
    #         rs = []
    #         for n in range(len(IRAF)):
    #             r, p = stats.pearsonr(IRAF[n], activity[n])
    #             r = np.arctanh(r)
    #             rs.append(r)
    #         t = np.mean(rs) / np.std(rs) * np.sqrt(len(rs))
    #         # if t > 3.0:
    #         #     print(f'POSITIVE {roi0} {roi1} {t=:.3f}')
    #         # elif t < -3.0:
    #         #     print(f'NEGATIVE {roi0} {roi1} {t=:.3f}')
    #         v.append(t)
    #     print(f'{len(v)=}')
    #     mat.append(v)
    # mat = np.array(mat)
    # print(mat.shape)
    title = 'YA. Activity x Object-IRAF, 1st-layer DNN.'
    plot_connectivity(t, ticks, tick_labels, tick_lows,
                      title=title, no_avg=True,
                      cbar_label='t-value')

def bulk_correlate(vals0, vals1):
    vals0 = np.expand_dims(vals0, axis=1)
    vals1 = np.expand_dims(vals1, axis=0)

    vals0_M = np.mean(vals0, axis=-1)
    vals0_SD = np.std(vals0, axis=-1)
    vals0_ = (vals0 - vals0_M[:, :, :, None]) / vals0_SD[:, :, :, None]
    vals1_M = np.mean(vals1, axis=-1)
    vals1_SD = np.std(vals1, axis=-1)
    vals1_ = (vals1 - vals1_M[:, :, :, None]) / vals1_SD[:, :, :, None]

    rs = vals0_ * vals1_
    rs = np.mean(rs, axis=-1)
    r_Ms = np.nanmean(rs, axis=-1)
    r_Ms[np.diag_indices_from(r_Ms)] = np.nan
    r_SDs = np.nanstd(rs, axis=-1)
    r_SDs[np.diag_indices_from(r_Ms)] = np.nan
    t = r_Ms / r_SDs * np.sqrt(rs.shape[-1]) # fix to account for different # nans per edge
    print(f'{t.shape=}')
    return r_Ms, r_SDs, t
    # print(rs.shape)
    # quit()
    # pass

def test_rxr():
    fp = get_cache_RSA_fp(cin=None, age=1, semantic=False,
                          early=True)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = get_ROI_info()
    data = d['rxr']['obj']
    data = np.array(data)
    # for val in data[:, 41, 45]:
    #     print(val > 0)

    # print(data[:, 41, 45])
    # quit()

    M = np.nanmean(data, axis=0)
    SD = np.nanstd(data, axis=0)
    # t = M
    t = M / SD * np.sqrt(len(data))

    # t[41, 45] = 100
    # print(t[41, 43])
    # quit()
    # t = np.mean(data > 0, axis=0)
    plot_connectivity(t, ticks, tick_labels, tick_lows,
                      no_avg=True,
                      # title='YA, 1st-layer DNN RSA for objects. '
                      #       'triple-correlation',
                      title='YA, 1st-layer DNN object RSA, '
                            'voxel x voxel connectivity ',
                      cbar_label='t-value')


    # print(np.array(data).shape)
    quit()


if __name__ == '__main__':
    # test_rxr()
    # CIN_compare()
    # analyze_ROIs()
    test_IRAF_x_activity(early=True, age=1)