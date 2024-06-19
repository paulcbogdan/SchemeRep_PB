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
t_end = time()
print(f'Import time: {t_end - t_st:.2f}')

config.CACHE_DIR = r'E:\PycharmProjects_E\SchemeRep\cache\numba'

NAN_VAL = 10001

CACHE_NUMBA = False

# sig = (nb.types.Tuple((nb.int32[:, :], nb.int32[:, :]))
#        (nb.boolean[:, :, :], nb.int32, nb.float64))


def convert_back_to_img(eval_results, mask, centers):
    img = np.full(mask.shape, NAN_VAL, dtype=np.float64)
    for i in range(centers.shape[0]):
        center = centers[i]
        img[center[0], center[1], center[2]] = eval_results[i]
    return img



@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def do_int_downsample(img, downsample, mask):
    img_smaller = np.zeros((img.shape[0] // downsample,
                            img.shape[1] // downsample,
                            img.shape[2] // downsample,
                            img.shape[3]))
    X_len_ = img_smaller.shape[0]
    Y_len_ = img_smaller.shape[1]
    Z_len_ = img_smaller.shape[2]
    n_samples = img.shape[3]
    size = downsample ** 3
    for x in range(X_len_):
        x_orig = x * downsample
        for y in range(Y_len_):
            y_orig = y * downsample
            for z in range(Z_len_):
                z_orig = z * downsample
                for n in range(n_samples):
                    vec = np.zeros(size)
                    cnt2 = 0
                    for dii in range(x_orig, x_orig + downsample):
                        for djj in range(y_orig, y_orig + downsample):
                            for dkk in range(z_orig, z_orig + downsample):
                                if mask[dii, djj, dkk]:
                                    vec[cnt2] = img[dii, djj, dkk, n]
                                    cnt2 += 1
                    if cnt2 == 0:
                        img_smaller[x, y, z, n] = NAN_VAL
                    else:
                        img_smaller[x, y, z, n] = np.sum(vec)# / cnt2

    return img_smaller

@jit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def do_int_upsample(img, upsample, mask):
    img_bigger = np.zeros((img.shape[0] * upsample,
                           img.shape[1] * upsample,
                           img.shape[2] * upsample,
                           ))
    X_len_ = img_bigger.shape[0]
    Y_len_ = img_bigger.shape[1]
    Z_len_ = img_bigger.shape[2]
    for x in range(X_len_):
        x_orig = x // upsample
        for y in range(Y_len_):
            y_orig = y // upsample
            for z in range(Z_len_):
                z_orig = z // upsample
                img_bigger[x, y, z] = img[x_orig, y_orig, z_orig]
    return img_bigger

def wrapped_jit_searchlight(sn, fp_fMRI_col, semantic, radius, skip_step,
                            downsample=1, resample=2, second_level='spear',
                            drop_prop=1):
    df_sn = get_trial_info(sn)
    df_sn, _ = sort_df_sn(df_sn, fp_fMRI_col)

    img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
    mask_pre = np.all(~np.isnan(img), axis=-1)

    if downsample != 1:
        img = do_int_downsample(img, downsample, mask_pre)

    img = np.transpose(img, (3, 0, 1, 2))
    print(f'{img.shape=}')

    mask = np.all(img != NAN_VAL, axis=0) & np.all(~np.isnan(img), axis=0)
    print(f'{mask.shape=}')
    t_st = time()
    centers, neighbors = jit_volume_searchlight(mask, radius=radius,
                                                threshold=0.25)

    t_end = time()
    print(f'Time needed to get neighbors: {t_end - t_st=:.2f}')

    t_st = time()
    data_2d = img.reshape([img.shape[0], -1])
    fMRI_RDMs = jit_searchlight_RDMs(data_2d, neighbors)
    t_end = time()
    t_taken = t_end - t_st
    print(f'Time needed to get RDMs: {t_taken=:.3f} s')

    t_st = time()
    d_vecs = prep_vecs(True, semantic)
    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True, dist='corr')
    tril_mask, kept_in = make_tril_mask_within_nan()
    RSM_stim_flat = RSM_stim[*tril_mask]
    fMRI_RDMs = fMRI_RDMs[:, kept_in]

    if second_level == 'spear':
        eval_results = evaluate_models_searchlight_spear(fMRI_RDMs,
                                                         RSM_stim_flat)
        # fMRI_RDMs = stats.rankdata(fMRI_RDMs, axis=1)
        # eval_results = evaluate_models_searchlight(fMRI_RDMs, RSM_stim_flat)
    else:
        eval_results = evaluate_models_searchlight(fMRI_RDMs, RSM_stim_flat)

    M_eval_results = np.nanmean(eval_results)
    print(f'{M_eval_results=:.3f}')


    t_end = time()
    t_taken = t_end - t_st
    print(f'Time needed to evaluate_models ({second_level}): {t_taken=:.3f} s')


    searched = convert_back_to_img(eval_results, mask, centers)

    if downsample != 1:
        searched = do_int_upsample(searched, downsample, mask)
    searched[searched == NAN_VAL] = np.nan
    mask = ~np.isnan(searched)

    return searched, ~mask

@cache
def make_tril_mask_within_nan():
    RSM = np.ones((114, 114))
    RSM = within_run_to_nan(RSM)
    tril_mask = ([], [])
    cnt = 0
    kept_in = np.zeros(114*113//2, dtype=np.bool_)
    for x in range(114): # matches in jit_searchlight_RDMs
        for y in range(x):
            if np.isnan(RSM[x, y]):
                cnt += 1
                continue
            tril_mask[0].append(x)
            tril_mask[1].append(y)
            kept_in[cnt] = True
            cnt += 1

    tril_mask_ = np.array(tril_mask)

    # for i, (x, y) in enumerate(zip(tril_mask[0], tril_mask[1])):
    #     if np.isnan(RSM[x, y]):
    #         continue
    #     tril_mask_[0].append(x)
    #     tril_mask_[1].append(y)
    #     kept_in[i] = True
    # tril_mask_ = np.array(tril_mask_)
    # for x, y in zip(*tril_mask_):
    #     # if np.isnan(RSM[x, y]):
    #     print(f'{x=}, {y=}')
    # quit()
    return tril_mask_, kept_in

def pad(data):
    good = np.isfinite(data)
    interpolated = np.interp(np.arange(data.shape[0]),
                             np.flatnonzero(good),
                             data[good])
    return interpolated

def interpolate_missing(array):
    from scipy import interpolate
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])

    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)

    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                               (xx, yy),
                               method='nearest')
    return GD1

def test_searchlight(semantic=True, radius=4, skip_step=1, downsample=1,
                     second_level='corr'):
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    fps = prep_fps('7')
    fps = ['obj7_fMRI']
    searched_all = []
    # sns = sns[2:]
    sns = sns[:30]
    t_last_plot = 0
    for sn in sns:
        sn_l = []
        for fp_fMRI_col in fps:
            dir_save = f'cache/searchlight_v2'
            Path(dir_save).mkdir(parents=True, exist_ok=True)
            fp_save = (f'{dir_save}/{sn}_{fp_fMRI_col}_{radius}_{semantic}_'
                       f'{skip_step}_{downsample}.pkl')
            t_st = time()
            kw = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
                  'radius': radius, 'skip_step': skip_step,
                  'downsample': downsample, 'second_level': second_level}
            searched, nan_mask = utils.pickle_wrap(
                lambda: wrapped_jit_searchlight(**kw), fp_save,
                verbose=-1, easy_override=True)

            max_val = np.nanmax(searched)
            min_val = np.nanmin(searched)
            print(f'{max_val=}, {min_val=}')
            subj_M = np.nanmean(searched)
            print(f'{sn}: {subj_M=:.3f}')
            t_end = time()
            if t_end - t_last_plot > 10:

                # vabs = np.nanmax(np.abs(searched[30, :, :]))
                vabs = 0.05
                plt.imshow(searched[30, :, :], vmin=-vabs, vmax=vabs,
                           cmap='cold_hot')
                plt.colorbar()
                plt.show()
                t_last_plot = t_end

            t_needed = t_end - t_st
            print(f'{t_needed=:.2f}, {searched.shape} | {kw=}')
            sn_l.append(searched)
            break
        searched = np.nanmean(sn_l, axis=0)
        searched_all.append(searched)

        # break
    searched_all = np.array(searched_all)
    print(searched_all.shape)
    # t, _ = stats.ttest_1samp(searched_all, 0, axis=0, nan_policy='omit')
    # t = np.array(t)
    # M = np.nanmean(searched_all, axis=0)
    M = np.nanmedian(searched_all, axis=0)

    SD = np.nanstd(searched_all, axis=0)
    N = np.sum(~np.isnan(searched_all), axis=0)
    SE = SD / np.sqrt(N)
    t = M / SE
    t_M = np.nanmean(t)
    print(f'{t_M=:.3f}')
    # print(t.shape)
    atlas = get_atlas()
    # print(atlas['maps'].shape)
    t_img = image.new_img_like(atlas['maps'], M)
    # M_img = image.new_img_like(atlas['maps'], M)
    # t_img = image.threshold_img(t_img, threshold=0.01, cluster_threshold=20)
    # plotting.plot_stat_map(t_img, threshold=0.01, vmin=-6, vmax=6)
    plotting.plot_stat_map(t_img,
                           display_mode="y",
                           vmin=-6, vmax=6, threshold=2
                           # cmap='turbo'
                           )#, threshold=0.01, vmin=-6, vmax=6)

    plt.show()

# from numba import njit, prange, set_num_threads
# set_num_threads(2)
t_end = time()
print(f'Startup time: {t_end - t_st:.2f}')
if __name__ == '__main__':
    test_searchlight()


