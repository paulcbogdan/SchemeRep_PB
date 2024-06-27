from time import time

from connRSA.jit_funcs import evaluate_models_searchlight_spear, evaluate_models_searchlight, jit_searchlight_RDMs, \
    jit_volume_searchlight
from connRSA.conn_utils import mask_img

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

def plot_t(t, title, vabs=None, fn='', only_positive=True, flip_color=False,
           dic='searchlight'):
    t_M = np.nanmean(t)

    print(f'{t_M=:.3f}')
    t_min = np.nanmin(t)
    print(f'\t{t_min=:.2f}')
    t_max = np.nanmax(t)
    print(f'\t{t_max=:.2f}')

    atlas = get_atlas()

    x_pre_pad = atlas['maps'].shape[0] - t.shape[0]
    y_pre_pad = atlas['maps'].shape[1] - t.shape[1]
    y_post_pad = y_pre_pad // 2
    y_pre_pad -= y_post_pad
    z_pre_pad = atlas['maps'].shape[2] - t.shape[2]
    t = np.pad(t, ((x_pre_pad, 0), (y_pre_pad, y_post_pad), (z_pre_pad, 0)))
    t = mask_img(t, None, blocks=True)

    if only_positive:
        t = np.maximum(t, 0)
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


    # t_img = image.threshold_img(t_img, threshold=3,
    #                             cluster_threshold=40)
    if vabs is None:
        vabs = np.nanquantile(np.abs(t), 0.995)


    fp_out = fr'E:\PycharmProjects_E\SchemeRep\result_pics\{dic}\{fn}.png'
    cmap = 'inferno' if only_positive else 'turbo'
    if flip_color:
        title = title.replace('Blue', 'XXX')
        title = title.replace('Red', 'Blue')
        title = title.replace('XXX', 'Red')
        cmap += '_r'
    plotting.plot_glass_brain(t_img, vmin=0 if only_positive else -vabs,
                              vmax=vabs, output_file=fp_out, plot_abs=False,
                              threshold=2, title=title,
                              colorbar=True, cmap=cmap,)

    plotting.plot_glass_brain(t_img, vmin=0 if only_positive else -vabs,
                              vmax=vabs, plot_abs=False,
                              threshold=2, title=title,
                              colorbar=True, cmap=cmap,
                              )
    plt.show()

    # plotting.plot_stat_map(t_img, display_mode='y',
    #                        cut_coords=tuple(range(-70, 20, 6)),
    #                        vmin=0 if only_positive else -vabs,
    #                        vmax=vabs,
    #                        threshold=2, title=title,
    #                        colorbar=True, cmap=cmap,
    #                        )
    #
    # plt.show()
    #
    # plotting.plot_stat_map(t_img, display_mode='mosaic',
    #                        vmin=0 if only_positive else -vabs,
    #                        vmax=vabs,
    #                        threshold=2, title=title,
    #                        colorbar=True, cmap=cmap,
    #                        )
    #
    # plt.show()
