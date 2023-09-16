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

def get_cache_RSA_fp(cin, age, semantic, early, pre_str=''):
    age_str = 'healthy' if age == 'healthy' else \
        'YA' if age == 1 else 'OA'
    cin_str = '' if cin is None else \
        '_Con' if cin == 1 else \
        '_Inc' if cin == 2 else \
        '_Neu' if cin == 3 else 'BAD_CIN'
    sem_str = '_sem' if semantic else ''
    el_str = '' if semantic else '_early' if early else '_late'
    fp_out = fr'cache/RSA/{age_str}{cin_str}{sem_str}{el_str}.pkl'
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

def test_IRAF_x_activity(age='healthy', early=True, semantic=False):
    fp = get_cache_RSA_fp(cin=None, age=age, semantic=semantic, early=early)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = get_ROI_info()

    mat = []
    for roi0 in ROIs:
        IRAF = d['IRAFs_ROI']['obj'][roi0]
        IRAF_m = np.mean(IRAF, axis=1)
        # print(f'{IRAF.shape=}')
        # print(f'{IRAF.shape=}')
        # quit()
        t_iraf = np.mean(IRAF_m, axis=0) / np.std(IRAF_m, axis=0) * np.sqrt(len(IRAF))
        if abs(t_iraf) > 2.0:
            print(f'IRAF {roi0} {t_iraf=:.3f}')
        # if IRAF.shape[0] != 56: continue
        # quit()
        v = []
        for roi1 in ROIs:
            # if roi0 != roi1: continue
            d['activity'][roi1] = np.array(d['activity'][roi1])
            activity = d['activity'][roi1]
            if activity.shape[0] != len(IRAF):
                v.append(np.nan)
                continue
            # quit()
            rs = []
            for n in range(len(IRAF)):
                r, p = stats.pearsonr(IRAF[n], activity[n])
                r = np.arctanh(r)
                rs.append(r)
            t = np.mean(rs) / np.std(rs) * np.sqrt(len(rs))
            if t > 3.0:
                print(f'POSITIVE {roi0} {roi1} {t=:.3f}')
            elif t < -3.0:
                print(f'NEGATIVE {roi0} {roi1} {t=:.3f}')
            v.append(t)
        print(f'{len(v)=}')
        mat.append(v)
    mat = np.array(mat)
    print(mat.shape)
    plot_connectivity(mat, ticks, tick_labels, tick_lows,
                      title='Activity x IRAF', no_avg=True)

def test_rxr():
    fp = get_cache_RSA_fp(cin=None, age=1, semantic=False, early=True)
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = get_ROI_info()
    data = d['rxr']['obj']
    M = np.nanmean(data, axis=0)
    SD = np.nanstd(data, axis=0)
    t = M / SD * np.sqrt(len(data))
    print(t.shape)
    plot_connectivity(t, ticks, tick_labels, tick_lows,
                      no_avg=True)


    # print(np.array(data).shape)
    quit()


if __name__ == '__main__':
    test_rxr()
    # CIN_compare()
    # test_IRAF_x_activity(early=False, age=2)