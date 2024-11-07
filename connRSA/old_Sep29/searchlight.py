from functools import cache
from pathlib import Path
from time import time

from numba import prange, set_num_threads, config
import numba as nb
from scipy.interpolate import RegularGridInterpolator

import utils
from Utils.atlas_funcs import get_atlas
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from org_sns import get_sns
from organize_bhv import get_trial_info
import numpy as np
import matplotlib.pyplot as plt

from stim import get_stim_RDM
from nilearn import plotting, image

config.CACHE_DIR = r'H:\PycharmProjects_H\SchemeRep\cache\numba'

NAN_VAL = 10001

def np_searchlight(img, nan_mask, radius, RSM, skip_step=1):
    X_len = img.shape[0]
    Y_len = img.shape[1]
    Z_len = img.shape[2]
    num_do = 0
    out = np.full((X_len, Y_len, Z_len), np.nan, dtype=np.float32)
    trils = np.tril_indices_from(RSM, k=-1)
    stim_flat = RSM[trils]

    size = (2 * radius + 1) ** 3

    # x_vals = np.array([x for x in range(radius, X_len - radius, skip_step)])
    for x in range(radius, X_len - radius):
        for y in range(radius, Y_len - radius, skip_step):
            for z in range(radius, Z_len - radius, skip_step):
                if nan_mask[x, y, z]:
                    continue
                square = img[x-radius:x+radius+1,
                             y-radius:y+radius+1,
                             z-radius:z+radius+1]
                square_flat = square.reshape((size, 114), order='F').T
                nans_flat = np.any(square_flat > 10000, axis=0)
                square_flat = square_flat[:, ~nans_flat]
                fMRI_RSM = np.corrcoef(square_flat)
                fMRI_flat = fMRI_RSM[trils]
                out[x, y, z] = np.corrcoef(stim_flat, fMRI_flat)[0, 1]
    return out


sig = nb.float64[:, :, :](nb.float64[:, :, :, :], nb.boolean[:, :, :], nb.int64,
                          nb.float64[:, :], nb.int32[:, :], nb.int64, nb.int64)

# @njit(sig, parallel=True, fastmath=True, nopython=True)#, cache=True)
def jit_searchlight(img, nan_mask, radius, RSM, tril_mask,
                    skip_step=1, downsample=1,):

    X_len = img.shape[0]
    Y_len = img.shape[1]
    Z_len = img.shape[2]
    num_do = 0
    out = np.full((X_len, Y_len, Z_len), np.nan, dtype=np.float32)
    # tril_mask = np.tril_indices_from(RSM, k=-1)

    stim_flat = np.full(4332, np.nan)
    for cnt in range(4332): # can't fancy index with numba???
        stim_flat[cnt] = RSM[tril_mask[0][cnt], tril_mask[1][cnt]]

    if downsample == 1:
        size = (2*radius+1)**3
    else:
        size = (2*(radius//downsample)+1)**3

    # downsample = 2

    # TODO: change to a sphere. Just predefine the sphere

    x_vals = np.array([x for x in range(radius, X_len-radius, skip_step)])
    x_vals_len = x_vals.shape[0]
    for x in prange(radius, X_len-radius):
        for y in range(radius, Y_len-radius, skip_step):
            for z in range(radius, Z_len-radius, skip_step):
                if nan_mask[x, y, z]:
                    continue

                square_flat = np.full((114, size), np.nan)
                cnt = 0


                if downsample == 1:
                    for di in range(-radius, radius+1):
                        for dj in range(-radius, radius+1):
                            for dk in range(-radius, radius+1):
                                square_flat[:, cnt] = img[x+di, y+dj, z+dk, :]
                                cnt += 1
                else:
                    for di in range(-radius, radius+1, downsample):
                        for dj in range(-radius, radius+1, downsample):
                            for dk in range(-radius, radius+1, downsample):
                                for n in range(114):
                                    vec = np.zeros(downsample ** 3)
                                    cnt2 = 0
                                    for dii in range(x+di, x+di+downsample):
                                        for djj in range(y+dj, y+dj+downsample):
                                            for dkk in range(z+dk, z+dk+downsample):
                                                if not nan_mask[dii, djj, dkk]:
                                                    vec[cnt2] = img[dii, djj, dkk, n]
                                                    cnt2 += 1
                                    if cnt2 == downsample ** 3:
                                        square_flat[n, cnt] = np.sum(vec) / cnt2
                                    else:
                                        square_flat[n, cnt] = NAN_VAL

                                cnt += 1

                # TODO: I could memoize this??
                nans_flat = np.full(size, False)
                for i in range(size): # NaNs dont work or something??
                    nans_flat[i] = np.any(square_flat[:, i] == NAN_VAL)
                square_flat = square_flat[:, ~nans_flat] # split spheres are fine
                # print(square_flat)
                fMRI_RSM = np.corrcoef(square_flat)
                fMRI_flat = np.full(4332, np.nan)
                for cnt in range(4332):
                    fMRI_flat[cnt] = fMRI_RSM[tril_mask[0][cnt],
                                              tril_mask[1][cnt]]

                out[x, y, z] = np.corrcoef(stim_flat, fMRI_flat)[0, 1]
                # plt.scatter(stim_flat, fMRI_flat)
                # plt.show()
                # quit()
                num_do += 1

    return out

def do_downsample(img, downsample, method='linear', target_shape=None):
    x = np.linspace(0, img.shape[0] - 1, img.shape[0], endpoint=True)
    y = np.linspace(0, img.shape[1] - 1, img.shape[1], endpoint=True)
    z = np.linspace(0, img.shape[2] - 1, img.shape[2], endpoint=True)
    interp = RegularGridInterpolator((x, y, z), img, method=method)

    if target_shape is not None:
        x_new = np.linspace(0, img.shape[0] - 1, target_shape[0], endpoint=True)
        y_new = np.linspace(0, img.shape[1] - 1, target_shape[1], endpoint=True)
        z_new = np.linspace(0, img.shape[2] - 1, target_shape[2], endpoint=True)
    else:
        x_new = np.linspace(0, img.shape[0] - 1, int(img.shape[0] / downsample),
                            endpoint=True)
        y_new = np.linspace(0, img.shape[1] - 1, int(img.shape[1] / downsample),
                            endpoint=True)
        z_new = np.linspace(0, img.shape[2] - 1, int(img.shape[2] / downsample),
                            endpoint=True)
    x_new, y_new, z_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    img = interp((x_new, y_new, z_new))
    return img




def wrapped_jit_searchlight(sn, fp_fMRI_col, semantic, radius, skip_step,
                            downsample=1, resample=2):
    df_sn = get_trial_info(sn)
    img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
    # if downsample != 1:
    #     img = do_downsample(img, downsample) # TODO: replace with nilearn
    # print(img.shape)
    # quit()

    img[np.isnan(img)] = NAN_VAL

    nans = np.all(img == NAN_VAL, axis=3)
    d_vecs = prep_vecs(True, semantic)
    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True, dist='corr')


    tril_mask = make_tril_mask_within_nan()

    kw = {'img': img, 'nan_mask': nans, 'radius': radius,
          'RSM': RSM_stim, 'skip_step': skip_step,
          'downsample': downsample, 'tril_mask': tril_mask
          }
    searched = jit_searchlight(**kw)
    # if downsample != 1:
    #     searched = do_downsample(searched, 1 / downsample, method='linear')
    return searched, nans

@cache
def make_tril_mask_within_nan(RSM_pre):
    RSM = np.ones((RSM_pre.shape[0], RSM_pre.shape[0]))
    RSM = within_run_to_nan(RSM)
    tril_mask = np.tril_indices_from(RSM, k=-1)
    tril_mask_ = ([], [])
    for x, y in zip(tril_mask[0], tril_mask[1]):
        if np.isnan(RSM[x, y]):
            continue
        tril_mask_[0].append(x)
        tril_mask_[1].append(y)
    tril_mask_ = np.array(tril_mask_)
    return tril_mask_

def pad(data):
    good = np.isfinite(data)
    interpolated = np.interp(np.arange(data.shape[0]),
                             np.flatnonzero(good),
                             data[good])
    return interpolated

def test_interpolate(array):
    from scipy import interpolate
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                               (xx, yy),
                               method='nearest')
    return GD1



def test_searchlight(semantic=True, radius=3, skip_step=1, downsample=1):
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    fps = prep_fps('7')
    searched_all = []
    # sns = sns[2:]
    sns = sns[:30]
    t_last_plot = 0
    for sn in sns:
        sn_l = []
        for fp_fMRI_col in fps:
            dir_save = f'cache/searchlight'
            Path(dir_save).mkdir(parents=True, exist_ok=True)
            fp_save = (f'{dir_save}/{sn}_{fp_fMRI_col}_{radius}_{semantic}_'
                       f'{skip_step}_{downsample}.pkl')

            # t_st = time()

            kw = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
                  'radius': radius, 'skip_step': skip_step,
                  'downsample': downsample}
            searched, nan_mask = Study1A.utils_pickle.pickle_wrap(
                lambda: wrapped_jit_searchlight(**kw), fp_save,
                verbose=-1, easy_override=False)

            for i in range(searched.shape[0]):
                searched[i] = test_interpolate(searched[i])
            searched[nan_mask] = np.nan

            t_end = time()
            if t_end - t_last_plot > 10:

                vabs = np.nanmax(np.abs(searched[30, :, :]))
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
    t_img = image.new_img_like(atlas['maps'], t)
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

if __name__ == '__main__':
    set_num_threads(2)

    test_searchlight()


