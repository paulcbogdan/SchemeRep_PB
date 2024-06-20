from time import time
import numpy as np
from tqdm import tqdm

import utils

from connRSA.jit_funcs import evaluate_regresslight, get_closest_dists, evaluate_regresslight_std
from connRSA.searchlight_plot import plot_t
from connRSA.searchlight_v2 import full_get_RDMs, NAN_VAL, results2img
from connRSA.single_trial_conn import prep_fps
from org_sns import get_sns
import matplotlib.pyplot as plt

def wrapped_jit_regresslight(sn, fp_fMRI_col, semantic,
                             radius1=2, downsample1=1, resample1=1,
                             radius2=2, downsample2=1, resample2=1,
                             second_level='spear', flip=False, std=True,
                             mask_ROIs=None):

    fMRI_RDMs1, RSM_stim_flat, mask, centers1, mask_downsample_pre = (
        full_get_RDMs(sn, fp_fMRI_col, semantic, flip=flip, radius=radius1,
                      downsample=downsample1, resample=resample1,
                      mask_ROIs=mask_ROIs))

    fMRI_RDMs2, _, _, centers2, _ = (
        full_get_RDMs(sn, fp_fMRI_col, semantic, flip=flip, radius=radius2,
                      downsample=downsample2, resample=resample2,
                      mask_ROIs=mask_ROIs))

    assert len(centers1) > len(centers2), f'{len(centers1)=} | {len(centers2)=}'

    t_st = time()
    idx0to2 = get_closest_dists(centers1.astype(np.float32),
                                centers2.astype(np.float32),
                                mult1=downsample1/downsample2)
    t_taken = time() - t_st
    print(f'Time needed to find closest: {t_taken:.3f} s')

    fMRI_RDMs2_in_1space = fMRI_RDMs2[idx0to2]
    assert fMRI_RDMs2_in_1space.shape == fMRI_RDMs1.shape
    assert np.sum(np.isnan(RSM_stim_flat)) == 0, \
        f'{np.sum(np.isnan(RSM_stim_flat))=}'

    t_st = time()
    if std:
        betas = evaluate_regresslight_std(fMRI_RDMs1, fMRI_RDMs2_in_1space,
                                          RSM_stim_flat)
    else:
        betas = evaluate_regresslight(fMRI_RDMs1, fMRI_RDMs2_in_1space,
                                      RSM_stim_flat)

    eval_results1 = betas[:, 1]
    eval_results2 = betas[:, 2]

    t_taken = time() - t_st
    print(f'Evaluate regresslight: {t_taken:.3f} s')

    searched1 = results2img(eval_results1, second_level, mask, centers1,
                            mask_downsample_pre, resample1, downsample1)

    searched2 = results2img(eval_results2, second_level, mask, centers1,
                            mask_downsample_pre, resample1, downsample1)
    mask = ~np.isnan(searched1)

    return searched1, searched2, mask

def test_regresslight(semantic=True, second_level='corr', flip=False,
                      radius1=2, downsample1=1, resample1=1,
                      radius2=2, downsample2=1, resample2=1,
                      std=True):
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    fps = prep_fps('7')
    # fps = ['obj7_fMRI']
    searched_all1 = []
    searched_all2 = []
    sns = sns[:4]
    # t_last_plot = 0
    for sn_i, sn in enumerate(sns):
        sn_l1 = []
        sn_l2 = []
        for fp_fMRI_col in fps:
            t_st = time()
            kw = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
                  'radius1': radius1, 'downsample1': downsample1,
                  'resample1': resample1,
                  'radius2': radius2, 'downsample2': downsample2,
                  'resample2': resample2,
                  'second_level': second_level, 'flip': flip, 'std': std}
            searched1, searched2, mask = utils.pickle_wrap(
                wrapped_jit_regresslight, kwargs=kw, verbose=-1,
                easy_override=False)

            # if t_end - t_last_plot > 60:
            # vabs = 0.05
            # plt.imshow(searched1[30, :, :], vmin=-vabs, vmax=vabs,
            #            cmap='cold_hot')
            # plt.colorbar()
            # plt.show()
            # plt.imshow(searched2[30, :, :], vmin=-vabs, vmax=vabs,
            #            cmap='cold_hot')
            # plt.colorbar()
            # plt.show()

            t_needed = time() - t_st
            print(f'{sn}, {fp_fMRI_col}: '
                  f'{t_needed=:.2f}, {searched1.shape} | {kw=}\n')
            sn_l1.append(searched1)
            sn_l2.append(searched2)
        searched1 = np.nanmean(sn_l1, axis=0)
        searched2 = np.nanmean(sn_l2, axis=0)
        searched_all1.append(searched1)
        searched_all2.append(searched2)


    searched_all1_ = np.array(searched_all1)
    searched_all2_ = np.array(searched_all2)
    t1 = get_t(searched_all1_)
    t2 = get_t(searched_all2_)
    plot_t(t1, title=f'Reg: {flip=}, {downsample1=}, {radius1=}')
    plot_t(t2, title=f'Reg: {flip=}, {downsample2=}, {radius2=}')


    # plot_t(t, flip, downsample, radius)

def get_t(searched_all):
    M = np.nanmedian(searched_all, axis=0)
    SD = np.nanstd(searched_all, axis=0)
    N = np.nansum(~np.isnan(searched_all), axis=0)
    SE = SD / np.sqrt(N)
    t = M / SE
    biggest_N = np.nanmax(N)
    t[N < int(biggest_N * 0.8)] = np.nan
    return t

if __name__ == '__main__':
    test_regresslight(semantic=True,
                      radius1=10, downsample1=1, resample1=10,
                      radius2=2, downsample2=5, resample2=10,)

    # test_regresslight(semantic=True,
    #                   radius1=8, downsample1=1, resample1=10,
    #                   radius2=8, downsample2=2, resample2=10,)