from time import time

from connRSA.jit_funcs import evaluate_models_searchlight_spear, evaluate_models_searchlight, jit_searchlight_RDMs, \
    jit_volume_searchlight

t_st = time()
from functools import cache
from pathlib import Path

from numba import jit, njit, prange, set_num_threads, config
import numba as nb
from scipy.interpolate import RegularGridInterpolator

import utils
from atlas_utils import get_atlas
from connRSA.searchlight import do_downsample
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from org_sns import get_sns
from organize_bhv import get_trial_info, sort_df_sn
import numpy as np
import matplotlib.pyplot as plt

from stim import get_stim_RDM
from nilearn import plotting, image
import scipy.stats as stats

def plot_t(t, title, vabs=None, fn=''):
    t_M = np.nanmean(t)

    print(f'{t_M=:.3f}')

    atlas = get_atlas()

    x_pre_pad = atlas['maps'].shape[0] - t.shape[0]
    y_pre_pad = atlas['maps'].shape[1] - t.shape[1]
    y_post_pad = y_pre_pad // 2
    y_pre_pad -= y_post_pad
    z_pre_pad = atlas['maps'].shape[2] - t.shape[2]
    t = np.pad(t, ((x_pre_pad, 0), (y_pre_pad, y_post_pad), (z_pre_pad, 0)))

    t_img = image.new_img_like(atlas['maps'], t)

    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    l_Ms = []
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        t_roi = t[atlas_roi]
        M_r_M = np.nanmean(t_roi)
        l_Ms.append(M_r_M)
    # plt.plot(l_Ms)
    # plt.xticks(atlas['ticks'], atlas['tick_labels'], rotation=45,
    #            fontsize=10)
    # plt.plot([0, len(atlas['ROIs'])], [0, 0], 'k--')
    # plt.show()


    # plotting.plot_stat_map(t_img, display_mode="x",
    #                        vmin=-tile, vmax=tile,
    #                        # vmin=-10, vmax=10, threshold=.01
    #                        )

    # t_img = image.threshold_img(t_img, threshold=3,
    #                             cluster_threshold=40)
    if vabs is None:
        vabs = np.nanquantile(np.abs(t), 0.99)

    fp_out = fr'E:\PycharmProjects_E\SchemeRep\result_pics\searchlight\{fn}.png'

    # 'cold_hot'
    plotting.plot_glass_brain(t_img, vmin=0, vmax=vabs, output_file=fp_out,
                              threshold=0, title=title,
                              colorbar=True, cmap='inferno',)
    plotting.plot_glass_brain(t_img, vmin=0, vmax=vabs, threshold=0,
                              title=title, colorbar=True, cmap='inferno',)

    plt.show()

