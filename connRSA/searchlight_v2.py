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

def wrapped_jit_searchlight(sn, fp_fMRI_col, semantic, radius,
                            downsample=1, resample=1, second_level='spear',
                            flip=False):
    df_sn = get_trial_info(sn)
    df_sn, _ = sort_df_sn(df_sn, fp_fMRI_col)

    img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
    mask_pre = np.all(~np.isnan(img), axis=-1)

    if downsample != 1:
        img = do_int_downsample(img, downsample, mask_pre)
    else:
        img[np.isnan(img)] = NAN_VAL

    mask_downsample_pre = np.all(img != NAN_VAL, axis=-1)
    img = np.transpose(img, (3, 0, 1, 2))
    print(f'{img.shape=}')

    mask = np.all(img != NAN_VAL, axis=0) & np.all(~np.isnan(img), axis=0)
    t_st = time()
    centers, neighbors = jit_volume_searchlight(mask, radius=radius,
                                                threshold=0.25)

    if resample > 1:
        idxs = np.arange(0, len(centers))
        idxs = np.random.choice(idxs, len(idxs) // resample, replace=False)
        centers = centers[idxs, :]
        neighbors = neighbors[idxs, :]

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

    t_st = time()
    d_vecs = prep_vecs(True, semantic)
    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True, dist='corr')
    if not flip: RSM_stim = within_run_to_nan(RSM_stim)
    tril_mask, kept_in = make_tril_mask_within_nan(flip=flip) # TODO: ???
    RSM_stim_flat = RSM_stim[*tril_mask]
    num_nans = np.sum(np.isnan(RSM_stim_flat))
    assert num_nans == 0, f'RSM stim flat still has nans{num_nans=}'
    fMRI_RDMs = fMRI_RDMs[:, kept_in]

    if second_level == 'spear':
        # for i in range(0, 10000, 1000):
        #     plt.scatter(fMRI_RDMs[i], RSM_stim_flat)
        #     plt.show()
        # quit()

        eval_results = evaluate_models_searchlight_spear(fMRI_RDMs,
                                                         RSM_stim_flat)
        # plt.hist(eval_results, bins=100)
        # M = np.mean(eval_results)
        # median = np.median(eval_results)
        # plt.title(f'{M=}, {median=}')
        # plt.show()
        # quit()
        # fMRI_RDMs = stats.rankdata(fMRI_RDMs, axis=1)
        # eval_results = evaluate_models_searchlight(fMRI_RDMs, RSM_stim_flat)
    else:
        eval_results = evaluate_models_searchlight(fMRI_RDMs, RSM_stim_flat)

    M_eval_results = np.nanmean(eval_results)
    print(f'{M_eval_results=:.3f}')


    t_end = time()
    t_taken = t_end - t_st
    print(f'Time needed to evaluate_models ({second_level}): {t_taken=:.3f} s')

    print(f'{mask_downsample_pre.shape=}')
    searched = convert_back_to_img(eval_results, mask, centers)
    print(f'{searched.shape=}')
    if resample > 1:
        searched = interpolate_nearest_3D(searched)
        searched[~mask_downsample_pre] = NAN_VAL

    if downsample != 1:
        searched = do_int_upsample(searched, downsample, mask)
    searched[searched == NAN_VAL] = np.nan
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

def test_searchlight(semantic=False, radius=2, downsample=1,
                     second_level='corr', resample=10, flip=False):
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    fps = prep_fps('7')
    # fps = ['obj7_fMRI']
    searched_all = []
    sns = sns[:20]
    t_last_plot = 0
    for sn in sns:
        sn_l = []
        for fp_fMRI_col in fps:
            # dir_save = f'cache/searchlight_v2'
            # Path(dir_save).mkdir(parents=True, exist_ok=True)
            # fp_save = (f'{dir_save}/{sn}_{fp_fMRI_col}_{radius}_{semantic}_'
            #            f'{skip_step}_{downsample}.pkl')
            t_st = time()
            kw = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
                  'radius': radius, #'skip_step': skip_step,
                  'downsample': downsample, 'second_level': second_level,
                  'resample': resample, 'flip': flip}
            # searched, nan_mask = utils.pickle_wrap(
            #     lambda: wrapped_jit_searchlight(**kw), fp_save,
            #     verbose=-1, easy_override=True)

            searched, nan_mask = utils.pickle_wrap(wrapped_jit_searchlight,
                                                   kwargs=kw, verbose=-1,
                                                   easy_override=False)

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
            print(f'{t_needed=:.2f}, {searched.shape} | {kw=}')
            sn_l.append(searched)
            # break
        searched = np.nanmean(sn_l, axis=0)
        searched_all.append(searched)

        # break
    searched_all = np.array(searched_all)
    print(searched_all.shape)
    # t, _ = stats.ttest_1samp(searched_all, 0, axis=0, nan_policy='omit')
    # t = np.array(t)
    # M = np.nanmean(searched_all, axis=0)
    M = np.nanmedian(searched_all, axis=0)
    M_M = np.nanmean(M)
    print(f'{M_M=:.6f}')

    SD = np.nanstd(searched_all, axis=0)
    N = np.nansum(~np.isnan(searched_all), axis=0)
    SE = SD / np.sqrt(N)
    t = M / SE
    biggest_N = np.nanmax(N)
    t[N < int(biggest_N * 0.8)] = np.nan
    t_M = np.nanmean(t)

    # plt.imshow(t[30, :, :])#, vmin=-0.05, vmax=0.05, cmap='cold_hot')
    # plt.colorbar()
    # plt.show()
    # quit()
    print(f'{t_M=:.3f}')
    # print(t.shape)
    atlas = get_atlas()
    # print(atlas['maps'].shape)

    x_pre_pad = atlas['maps'].shape[0] - t.shape[0]
    y_pre_pad = atlas['maps'].shape[1] - t.shape[1]
    y_post_pad = y_pre_pad // 2
    y_pre_pad -= y_post_pad
    z_pre_pad = atlas['maps'].shape[2] - t.shape[2]
    t = np.pad(t, ((x_pre_pad, 0), (y_pre_pad, y_post_pad), (z_pre_pad, 0)))

    # for dim in range(3): # pad
    #     if t.shape[dim] < atlas['maps'].shape[dim]:
    #         t = np.pad(t, 0)


            # t = np.pad(t, ((0, atlas['maps'].shape[dim] - t.shape[dim]),
            #                (0, 0), (0, 0)))


    t_img = image.new_img_like(atlas['maps'], t)
    # t = t_img.get_fdata()
    # M_img = image.new_img_like(atlas['maps'], M)
    # t_img = image.threshold_img(t_img, threshold=0.01, cluster_threshold=20)
    # plotting.plot_stat_map(t_img, threshold=0.01, vmin=-6, vmax=6)

    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    ROI2vecs = {}
    # region2vecs = defaultdict(list)
    l_Ms = []
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num

        t_roi = t[atlas_roi]
        M_r_M = np.nanmean(t_roi)
        l_Ms.append(M_r_M)
    plt.plot(l_Ms)
    plt.xticks(atlas['ticks'], atlas['tick_labels'], rotation=45,
               fontsize=10)
    plt.plot([0, len(atlas['ROIs'])], [0, 0], 'k--')
    plt.show()

    tile = np.nanquantile(t, 0.999)
    print(f'{tile=}')

    # plotting.plot_stat_map(t_img, display_mode="x",
    #                        vmin=-tile, vmax=tile,
    #                        # vmin=-10, vmax=10, threshold=.01
    #                        )

    t_img = image.threshold_img(t_img, threshold=2,
                                cluster_threshold=40)

    plotting.plot_glass_brain(t_img, display_mode="x",
                              vmin=-tile, vmax=tile, colorbar=True,
                              cmap='cold_hot',)

    plt.title(f'{flip=}, {downsample=}, {radius=}')
    plt.show()
    quit()


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

from numba import njit, prange, set_num_threads
set_num_threads(2)
t_end = time()


print(f'Startup time: {t_end - t_st:.2f}')
if __name__ == '__main__':
    for RADIUS in [8, 12]:
        test_searchlight(downsample=2, radius=RADIUS, flip=False)


