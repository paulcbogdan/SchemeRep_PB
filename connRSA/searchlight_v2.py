import os
from time import time

from matplotlib import pyplot as plt, image as mpimg

from Utils.atlas_funcs import get_atlas
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.conn_utils import mask_img
from connRSA.jit_funcs import evaluate_models_searchlight_spear, evaluate_models_searchlight, jit_searchlight_RDMs, \
    jit_volume_searchlight, do_int_downsample, do_int_upsample, convert_back_to_img, prep_data_Ms
from connRSA.searchlight_plot import plot_t
from networks.old.networks import prep_networks

# from old.networks import prep_networks

t_st = time()
from functools import cache

from numba import config, set_num_threads

import utils
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from org_sns import get_sns
from organize_bhv import get_trial_info, sort_df_sn
import numpy as np

from stim import get_stim_RDM

t_end = time()
print(f'Import time: {t_end - t_st:.2f}')

config.CACHE_DIR = r'C:\PycharmProjects\SchemeRep\cache\numba'

NAN_VAL = 10000001

CACHE_NUMBA = False

np.random.seed(0)

def get_singular_ROIs(target_ROI):
    regions = set(prep_networks(
        network_setting=ROI2NETWORK[target_ROI])[target_ROI])
    atlas = get_atlas()
    ROIs_match = []
    for ROI in atlas['ROIs']:
        for region in regions:
            if region in ROI:
                ROIs_match.append(ROI)
                break
    return ROIs_match


def full_get_RDMs(sn, fp_fMRI_col, radius, downsample=1,
                  resample=1, flip=False, mask_ROIs=None,
                  resample_filterer=None, threshold=0.25,
                  get_sphere_Ms=False, no_RDMs=False,
                  get_vox2center=False, cube=True): # first_level='corr'
    df_sn = get_trial_info(sn)
    df_sn, _ = sort_df_sn(df_sn, fp_fMRI_col)

    img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
    if mask_ROIs is not None:
        img = mask_img(img, mask_ROIs)
    else:
        img = mask_img(img, None)

    mask_pre = np.all(~np.isnan(img), axis=-1)

    if downsample != 1:
        img = do_int_downsample(img, downsample, mask_pre)
    else:
        img[np.isnan(img)] = NAN_VAL

    # mask_pre = np.all(~np.isnan(img), axis=-1)
    mask_downsample_pre = np.all(img != NAN_VAL, axis=-1)
    img = np.transpose(img, (3, 0, 1, 2))
    t_st = time()
    img = img.astype(np.float32)

    print(f'\tTime to change to np.float32: {time() - t_st:.2f} s')

    mask = np.all(img != NAN_VAL, axis=0) & np.all(~np.isnan(img), axis=0)
    t_st = time()
    data_2d = img.reshape([img.shape[0], -1])
    del img
    assert np.sum(np.isnan(data_2d)) == 0, 'data_2d has nans'

    voxels_in_max = np.sum(mask)
    print(f'{voxels_in_max=}')
    centers, neighbors, voxel2idx_centers = (
        jit_volume_searchlight(mask, radius=radius, threshold=threshold,
                               resample=resample, cube=cube))
    num_valid_spots = neighbors.shape[0]
    # for threshold in [.05, .1, .15, .2, .25, .3]:
    #     centers, neighbors, voxel2idx_centers = (
    #         jit_volume_searchlight(mask, radius=radius, threshold=threshold,
    #                                resample=resample, cube=cube))
    #     num_valid_spots = neighbors.shape[0]
    print(f'{threshold}: {num_valid_spots/voxels_in_max=:.1%}')

    if no_RDMs:
        if centers.shape[0] == 0:
            return [None] * 3
        else:
            return centers, mask, mask_downsample_pre
    assert np.sum(np.isnan(centers)) == 0, 'centers has nans'
    assert np.sum(np.isnan(neighbors)) == 0, 'neighbors has nans'
    if centers.shape[0] == 0:
        print(f'NO CENTERS: {sn}/{fp_fMRI_col}')
        if get_vox2center:
            return [None] * 6
        elif get_sphere_Ms:
            return [None] * 5
        else:
            return [None] * 4

    print(f'\t\tNumber of centers: {len(centers)}, {neighbors.shape=}')

    assert (np.max(neighbors) < 1e7 or np.max(neighbors) == NAN_VAL), \
        f'{np.max(neighbors)=}'
    t_end = time()
    print(f'\tTime needed to get neighbors: {t_end - t_st:.2f} s')


    assert np.sum(np.isnan(neighbors)) == 0, 'neighbors has nans'
    t_st = time()
    fMRI_RDMs = jit_searchlight_RDMs(data_2d, neighbors)
    print(f'\tTime needed to get RDMs: {time() - t_st:.2f} s')
    if get_sphere_Ms:
        t_st = time()
        print(f'{data_2d.shape=}')
        data_Ms = prep_data_Ms(data_2d, neighbors)

        print(f'\tTime needed to get data_Ms: {time() - t_st:.2f} s')

    del data_2d, neighbors
    assert np.sum(np.isnan(fMRI_RDMs)) == 0, 'fMRI_RDMs has nans'



    tril_mask, kept_in = make_tril_mask_within_nan(flip=flip)
    fMRI_RDMs = fMRI_RDMs[:, kept_in]

    if get_vox2center:
        return (fMRI_RDMs, mask, centers, mask_downsample_pre, data_Ms,
                voxel2idx_centers)

    elif get_sphere_Ms:
        return fMRI_RDMs, mask, centers, mask_downsample_pre, data_Ms
    else:
        return fMRI_RDMs, mask, centers, mask_downsample_pre

def get_RSM_stim_flat(semantic, sn, fp_fMRI_col, flip):
    df_sn = get_trial_info(sn)
    df_sn, _ = sort_df_sn(df_sn, fp_fMRI_col)
    d_vecs = prep_vecs(True, semantic)
    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True, dist='corr')
    # print(RSM_stim)
    if not flip: RSM_stim = within_run_to_nan(RSM_stim)
    tril_mask, kept_in = make_tril_mask_within_nan(flip=flip)
    RSM_stim_flat = RSM_stim[*tril_mask]
    num_nans = np.sum(np.isnan(RSM_stim_flat))
    assert num_nans == 0, f'RSM stim flat still has nans{num_nans=}'
    return RSM_stim_flat


def results2img(eval_results, second_level, mask, centers, mask_downsample_pre,
                resample, downsample):
    M_eval_results = np.nanmean(eval_results)
    t_st = time()

    searched = convert_back_to_img(eval_results, mask, centers)
    if resample > 1:
        searched = interpolate_nearest_3D(searched)
        searched[~mask_downsample_pre] = NAN_VAL
    print(f'\tTime needed to interpolate: {time() - t_st:.2f} s')

    t_st = time()
    if downsample != 1:
        searched = do_int_upsample(searched, downsample, mask)
    print(f'\tTime needed to upscale: {time() - t_st:.2f} s')

    searched[searched == NAN_VAL] = np.nan
    return searched


def wrapped_jit_searchlight(sn, fp_fMRI_col, semantic, radius,
                            downsample=1, resample=1, second_level='spear',
                            flip=False, mask_ROIs=None,
                            threshold=0.25, cube=False):

    # fMRI_RDMs, RSM_stim_flat, mask, centers, mask_downsample_pre = (
    #     full_get_RDMs(sn, fp_fMRI_col, semantic, radius, downsample=downsample,
    #                   resample=resample, flip=flip, mask_ROIs=mask_ROIs))

    # kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
    #        'flip': flip, 'radius': radius, 'downsample': downsample,
    #        'resample': resample, 'mask_ROIs': mask_ROIs}
    # fMRI_RDMs, RSM_stim_flat, mask, centers, mask_downsample_pre = (
    #     utils.pickle_wrap(full_get_RDMs, kwargs=kw1, verbose=0))
    # if fMRI_RDMs is None:
    #     return None, None

    kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col,
           'flip': flip, 'radius': radius, 'downsample': downsample,
           'resample': resample, 'mask_ROIs': mask_ROIs,
           'threshold': threshold, 'cube': cube}
    # fMRI_RDMs1, mask, centers1, mask_downsample_pre = (
    #     utils.pickle_wrap(full_get_RDMs, kwargs=kw1, verbose=0,
    #                       easy_override=False))

    fMRI_RDMs1, mask, centers1, mask_downsample_pre = full_get_RDMs(**kw1)
    if fMRI_RDMs1 is None:
        return None, None

    kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
           'flip': flip,}
    RSM_stim_flat = utils.pickle_wrap(get_RSM_stim_flat, kwargs=kw1,
                                      verbose=0)

    t_st = time()
    if second_level == 'spear':
        eval_results = evaluate_models_searchlight_spear(fMRI_RDMs1,
                                                         RSM_stim_flat)
    else:
        eval_results = evaluate_models_searchlight(fMRI_RDMs1, RSM_stim_flat)
    t_end = time()
    t_taken = t_end - t_st
    print(f'\tTime needed to evaluate_models ({second_level}): '
          f'{t_taken=:.2f} s\n')
    searched = results2img(eval_results, second_level, mask, centers1,
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
                     second_level='corr', resample=10, flip=False,
                     network=None, do_con=True, threshold=0.25,
                     cube=False):
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    fps = prep_fps('7')
    searched_all = []

    flip_str = ' flip' if flip else ''
    flip_str_ = '_flip' if flip else ''
    semantic_str = 'semantic' if semantic else 'visual'
    network_str = f'_{network}' if network is not None else ''
    sample_size = (60, 4 if do_con else 3)
    title =  (f'Corr, {semantic_str}, N={sample_size}. '
              f'Down: {downsample}, radius: {radius} ') + flip_str
    fn = (f'd{downsample}-r{radius}{flip_str_}_'
          f'{semantic_str}_{sample_size}{network_str}_thr{threshold}_'
          f'searchlight_r{resample}')
    dic = 'big_cube' if cube else 'searchlight'
    fp3 = rf'result_pics/{dic}/{fn}.png'
    if os.path.isfile(fp3) and False:
        plot_searchlight_fn(fn)
        return

    if network is not None:
        mask_ROIs = get_singular_ROIs(network)
    else:
        mask_ROIs = None

    base_shape = None
    # sns = sns[3::4]
    # sns = sns[:45]
    sns = sns[::-1]
    for sn in sns:
        sn_l = []
        for fp_fMRI_col in fps:
            t_st = time()
            kw = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
                  'radius': radius, 'downsample': downsample,
                  'second_level': second_level, 'resample': resample,
                  'flip': flip, 'mask_ROIs': mask_ROIs,
                  'threshold': threshold, 'cube': cube}

            searched, nan_mask = utils.pickle_wrap(wrapped_jit_searchlight,
                                                   kwargs=kw, verbose=-1,
                                                   easy_override=False)

            if searched is None:
                assert base_shape is not None
                print(f'\tNONE NONE NONE: {sn}, {fp_fMRI_col}')
                searched = np.full(base_shape, np.nan)
            else:
                assert base_shape is None or searched.shape == base_shape
                base_shape = searched.shape

            print(f'Total searchlight time: {time() - t_st:.2f} s, {searched.shape} '
                  f'| {kw=}')
            sn_l.append(searched)
        searched_all.append(sn_l)
    searched_all_ = np.array(searched_all)
    sample_size = searched_all_.shape[:2]
    title =  (f'Corr, {semantic_str}, N={sample_size}. '
              f'Down: {downsample}, radius: {radius} ') + flip_str
    fn = (f'd{downsample}-r{radius}{flip_str_}_'
          f'{semantic_str}_{sample_size}{network_str}_thr{threshold}_'
          f'searchlight')

    searched_all_ = np.nanmean(searched_all_, axis=1)
    M = np.nanmedian(searched_all_, axis=0)
    SD = np.nanstd(searched_all_, axis=0)
    N = np.nansum(~np.isnan(searched_all_), axis=0)
    SE = SD / np.sqrt(N)
    t = M / SE
    biggest_N = np.nanmax(N)
    t[N < int(biggest_N * 0.8)] = np.nan

    vabs = 8
    plot_t(t, title=title, fn=fn, vabs=vabs, thresh=0)



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

def plot_searchlight_fn(fn, dic='searchlight'):
    fp = rf'result_pics/{dic}/{fn}.png'
    fig = plt.figure(figsize=(7.5, 3.5))
    fig.subplots_adjust(bottom=0., left=0., right=1., top=1.)

    img = mpimg.imread(fp)
    plt.imshow(img)
    plt.axis('off')
    plt.show()



print(f'Startup time: {t_end - t_st:.2f}')
if __name__ == '__main__':
    set_num_threads(1)
    t_end = time()

    THRESHOLD = 0.2
    # THRESHOLD = 0.1
    for NETWORK in ['cortex', 'OC_IT', ]:
        for SEMANTIC in [True, False]:
            # SEMANTIC = False
            # test_searchlight(radius=18, downsample=1, flip=False,
            #                  semantic=SEMANTIC, resample=10,
            #                  network=NETWORK, threshold=THRESHOLD,
            #                  cube=True)
            # test_searchlight(radius=2, downsample=9, flip=False,
            #                  semantic=SEMANTIC, resample=10,
            #                  network=NETWORK, threshold=THRESHOLD,
            #                  cube=True)
            test_searchlight(radius=6, downsample=1, flip=False,
                                 semantic=SEMANTIC, resample=10,
                                 network=NETWORK, threshold=THRESHOLD,
                                 cube=True)


