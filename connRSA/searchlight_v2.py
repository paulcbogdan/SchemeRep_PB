from time import time

from atlas_utils import get_atlas
from connRSA.jit_funcs import evaluate_models_searchlight_spear, evaluate_models_searchlight, jit_searchlight_RDMs, \
    jit_volume_searchlight, do_int_downsample, do_int_upsample, convert_back_to_img
from connRSA.searchlight_plot import plot_t

t_st = time()
from functools import cache

from numba import config

import utils
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from org_sns import get_sns
from organize_bhv import get_trial_info, sort_df_sn
import numpy as np

from stim import get_stim_RDM

t_end = time()
print(f'Import time: {t_end - t_st:.2f}')

config.CACHE_DIR = r'E:\PycharmProjects_E\SchemeRep\cache\numba'

NAN_VAL = 10001

CACHE_NUMBA = False

def get_ROIs():
    pass

def mask_img(img, ROIs):
    t_st = time()
    ROIs = set(ROIs)
    atlas = get_atlas()
    img[atlas == 0] = np.nan # Needed to basically get only ROIs and no non-ROI
    for ROI, ROI_num in zip(atlas['ROIs'], atlas['ROI_nums']):
        if ROI not in ROIs:
            img[img == ROI_num] = np.nan
    print(f'Time needed to mask ROIs: {time() - t_st=:.2f}')
    return img

def full_get_RDMs(sn, fp_fMRI_col, semantic, radius, downsample=1,
                  resample=1, flip=False, mask_ROIs=None):
    df_sn = get_trial_info(sn)
    df_sn, _ = sort_df_sn(df_sn, fp_fMRI_col)

    img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
    if mask_ROIs is not None:
        img = mask_img(img, mask_ROIs)

    mask_pre = np.all(~np.isnan(img), axis=-1)

    if downsample != 1:
        img = do_int_downsample(img, downsample, mask_pre)
    else:
        img[np.isnan(img)] = NAN_VAL

    mask_downsample_pre = np.all(img != NAN_VAL, axis=-1)
    img = np.transpose(img, (3, 0, 1, 2))

    mask = np.all(img != NAN_VAL, axis=0) & np.all(~np.isnan(img), axis=0)
    t_st = time()
    centers, neighbors = jit_volume_searchlight(mask, radius=radius,
                                                threshold=0.25)
    if resample > 1:
        idxs = np.arange(0, len(centers))
        idxs = np.random.choice(idxs, len(idxs) // resample, replace=False)
        centers = centers[idxs, :]
        neighbors = neighbors[idxs, :]
    print(f'Number of centers: {len(centers)}, {neighbors.shape=}')

    if len(neighbors) == 0:
        return None, None
    assert np.max(neighbors) < 1e7
    t_end = time()
    print(f'Time needed to get neighbors: {t_end - t_st=:.2f}')

    t_st = time()
    data_2d = img.reshape([img.shape[0], -1])
    fMRI_RDMs = jit_searchlight_RDMs(data_2d, neighbors)
    t_end = time()
    t_taken = t_end - t_st
    print(f'Time needed to get RDMs: {t_taken=:.3f} s')

    d_vecs = prep_vecs(True, semantic)
    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True, dist='corr')
    if not flip: RSM_stim = within_run_to_nan(RSM_stim)
    tril_mask, kept_in = make_tril_mask_within_nan(flip=flip)
    RSM_stim_flat = RSM_stim[*tril_mask]
    num_nans = np.sum(np.isnan(RSM_stim_flat))
    assert num_nans == 0, f'RSM stim flat still has nans{num_nans=}'
    fMRI_RDMs = fMRI_RDMs[:, kept_in]

    return fMRI_RDMs, RSM_stim_flat, mask, centers, mask_downsample_pre

def results2img(eval_results, second_level, mask, centers, mask_downsample_pre,
                resample, downsample):
    M_eval_results = np.nanmean(eval_results)
    searched = convert_back_to_img(eval_results, mask, centers)
    if resample > 1:
        searched = interpolate_nearest_3D(searched)
        searched[~mask_downsample_pre] = NAN_VAL

    if downsample != 1:
        searched = do_int_upsample(searched, downsample, mask)
    searched[searched == NAN_VAL] = np.nan
    return searched


def wrapped_jit_searchlight(sn, fp_fMRI_col, semantic, radius,
                            downsample=1, resample=1, second_level='spear',
                            flip=False):

    fMRI_RDMs, RSM_stim_flat, mask, centers, mask_downsample_pre = (
        full_get_RDMs(sn, fp_fMRI_col, semantic, radius, downsample=downsample,
                      resample=resample, flip=flip))

    t_st = time()
    if second_level == 'spear':
        eval_results = evaluate_models_searchlight_spear(fMRI_RDMs,
                                                         RSM_stim_flat)
    else:
        eval_results = evaluate_models_searchlight(fMRI_RDMs, RSM_stim_flat)
    t_end = time()
    t_taken = t_end - t_st
    print(f'Time needed to evaluate_models ({second_level}): '
          f'{t_taken=:.3f} s\n')
    searched = results2img(eval_results, second_level, mask, centers,
                           mask_downsample_pre, resample, downsample)

    mask = ~np.isnan(searched)
    return searched, ~mask


@cache
def make_tril_mask_within_nan(flip=False):
    RSM = np.ones((114, 114))
    RSM = within_run_to_nan(RSM)
    tril_mask = ([], [])
    cnt = 0
    kept_in = np.zeros(114*113//2, dtype=np.bool_)
    for x in range(114): # matches in jit_searchlight_RDMs
        for y in range(x):
            if flip:
                if np.isnan(RSM[x, y]):
                    tril_mask[0].append(x)
                    tril_mask[1].append(y)
                    kept_in[cnt] = True
                cnt += 1
            else:
                if np.isnan(RSM[x, y]):
                    cnt += 1
                    continue
                tril_mask[0].append(x)
                tril_mask[1].append(y)
                kept_in[cnt] = True
                cnt += 1

    tril_mask_ = np.array(tril_mask)
    return tril_mask_, kept_in


def test_searchlight(semantic=False, radius=2, downsample=1,
                     second_level='corr', resample=10, flip=False):
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    fps = prep_fps('7')
    # fps = ['obj7_fMRI']
    searched_all = []
    # sns = sns[:20]
    for sn in sns:
        sn_l = []
        for fp_fMRI_col in fps:
            t_st = time()
            kw = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
                  'radius': radius, #'skip_step': skip_step,
                  'downsample': downsample, 'second_level': second_level,
                  'resample': resample, 'flip': flip}
            searched, nan_mask = utils.pickle_wrap(wrapped_jit_searchlight,
                                                   kwargs=kw, verbose=-1,
                                                   easy_override=True)
            if searched is None: continue
            max_val = np.nanmax(searched)
            min_val = np.nanmin(searched)
            print(f'{max_val=}, {min_val=}')
            subj_M = np.nanmean(searched)
            print(f'{sn}: {subj_M=:.3f}')
            t_end = time()
            # if t_end - t_last_plot > 60:
            #     vabs = 0.05
            #     plt.imshow(searched[30, :, :], vmin=-vabs, vmax=vabs,
            #                cmap='cold_hot')
            #     plt.colorbar()
            #     plt.show()
            #     t_last_plot = t_end
            t_needed = t_end - t_st
            print(f'Total searchlight time: {t_needed=:.2f}, {searched.shape} '
                  f'| {kw=}')
            # quit()
            sn_l.append(searched)
        searched = np.nanmean(sn_l, axis=0)
        searched_all.append(searched)
    searched_all_ = np.array(searched_all)
    M = np.nanmedian(searched_all_, axis=0)
    SD = np.nanstd(searched_all_, axis=0)
    N = np.nansum(~np.isnan(searched_all_), axis=0)
    SE = SD / np.sqrt(N)
    t = M / SE
    biggest_N = np.nanmax(N)
    t[N < int(biggest_N * 0.8)] = np.nan

    plot_t(t, title=f'{flip=}, {downsample=}, {radius=}')


def interpolate_nearest_3D(ar):
    from scipy import interpolate
    # val = ar.ravel()
    x = np.arange(0, ar.shape[1])
    y = np.arange(0, ar.shape[0])
    z = np.arange(0, ar.shape[2])
    X, Y, Z = np.meshgrid(x, y, z)

    x_pre, y_pre, z_pre = [], [], []
    vals = []
    for i in range(0, ar.shape[0]):
        for j in range(0, ar.shape[1]):
            for k in range(0, ar.shape[2]):
                if not np.isnan(ar[i, j, k]) and ar[i, j, k] != NAN_VAL:
                    x_pre.append(j)
                    y_pre.append(i)
                    z_pre.append(k)
                    vals.append(ar[i, j, k])

    interp = interpolate.NearestNDInterpolator((x_pre, y_pre, z_pre), vals)

    val_interp = interp(X, Y, Z)
    return val_interp

from numba import set_num_threads
set_num_threads(2)
t_end = time()


print(f'Startup time: {t_end - t_st:.2f}')
if __name__ == '__main__':
    for RADIUS in [3]:
        test_searchlight(downsample=1, radius=RADIUS, flip=False,
                         semantic=True, resample=10)


