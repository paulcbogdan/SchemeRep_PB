import os.path
from collections import defaultdict
from time import time
import numpy as np
from matplotlib import pyplot as plt

import utils

from connRSA.jit_funcs import evaluate_regresslight, get_closest_dists, evaluate_regresslight_std, \
    evaluate_multilight_std, evaluate_models_searchlight
from connRSA.searchlight_plot import plot_t
from connRSA.searchlight_v2 import full_get_RDMs, results2img, get_singular_ROIs, get_RSM_stim_flat, \
    plot_searchlight_fn
from connRSA.conn_utils import mask_img
from connRSA.single_trial_conn import prep_fps
from org_sns import get_sns
from numba import set_num_threads

def wrapped_jit_eightlight(sn, fp_fMRI_col, semantic,
                           radius1=2, downsample1=1, resample1=1,
                           radius2=2, downsample2=1, resample2=1,
                           second_level='spear', flip=False, std=True,
                           mask_ROIs=None, ):
    print()
    kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
           'flip': flip,}
    RSM_stim_flat = utils.pickle_wrap(get_RSM_stim_flat, kwargs=kw1,
                                      verbose=0, easy_override=False)
    assert np.sum(np.isnan(RSM_stim_flat)) == 0, \
        f'{np.sum(np.isnan(RSM_stim_flat))=}'

    kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col,
           'flip': flip, 'radius': radius1, 'downsample': downsample1,
           'resample': resample1, 'mask_ROIs': mask_ROIs}
    fMRI_RDMs1, mask, centers1, mask_downsample_pre = (
        utils.pickle_wrap(full_get_RDMs, kwargs=kw1, verbose=0,
                          easy_override=False))

    # fMRI_RDMs1, mask, centers1, mask_downsample_pre = full_get_RDMs(**kw1)

    if fMRI_RDMs1 is None:
        return None, None, None

    kw2 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col,
           'flip': flip, 'radius': radius2, 'downsample': downsample2,
           'resample': resample2, 'mask_ROIs': mask_ROIs,}

    fMRI_RDMs2, _, centers2, _ = utils.pickle_wrap(full_get_RDMs,
                                                   kwargs=kw2, verbose=0,
                                                   easy_override=False)

    fMRI_RDMs2, _, centers2, _ = full_get_RDMs(**kw2)

    print(f'{centers1.shape=}')
    print(f'{centers2.shape=}')
    if fMRI_RDMs2 is None:
        return None, None, None

    prop_nan1 = np.sum(np.isnan(fMRI_RDMs1)) / fMRI_RDMs1.size
    prop_nan2 = np.sum(np.isnan(fMRI_RDMs2)) / fMRI_RDMs2.size
    assert prop_nan1 == 0, f'{prop_nan1=}'
    assert prop_nan2 == 0, f'{prop_nan2=}'

    t_st = time()
    mult1 = downsample1 / downsample2

    idx1to_center = get_closest_dists(centers1.astype(np.float32),
                                      centers2.astype(np.float32),
                                      mult1=mult1)
    print(f'\tTime needed to find closest: {time() - t_st:.2f} s')

    off = int(radius1*2/3)
    print(off)
    i2new_spot = defaultdict(list)
    mod2idxs = {}
    for x_mod in [-off, 0, off]:
        for y_mod in [-off, 0, off]:
            for z_mod in [-off, 0, off]:
                num_zeros = np.sum([x_mod == 0, y_mod == 0, z_mod == 0])
                if num_zeros != 2:
                    continue
                mod2idxs[(np.sign(x_mod), np.sign(y_mod), np.sign(z_mod))] = (
                    get_closest_dists(centers1.astype(np.float32) +
                                      [x_mod, y_mod, z_mod],
                                      centers2.astype(np.float32),
                                      mult1=mult1))

                for i in range(len(centers1)):
                    i2new_spot[i].append(
                        (centers1[i] + [x_mod, y_mod, z_mod]))
    print(f'{centers1.shape=}')
    print(f'{centers2.shape=}')
    quit()
    # print(mod2idxs)
    ar = np.vstack([l for l in mod2idxs.values()]).T
    nan_out = []
    for i in range(ar.shape[0]):
        num_unique = len(np.unique(ar[i]))
        if num_unique != 6:
            nan_out.append(i)

    print(f'Number of non perfectly unique: {len(nan_out)}')

    ar = np.concatenate([idx1to_center[:, None], ar], axis=1)

    regressors = fMRI_RDMs2[ar]

    del fMRI_RDMs2
    assert regressors[:, 0, :].shape == fMRI_RDMs1.shape
    assert np.sum(np.isnan(regressors)) == 0, \
        f'{np.sum(np.isnan(regressors))=}'

    t_st = time()


    betas = evaluate_multilight_std(fMRI_RDMs1, regressors, RSM_stim_flat)
    eval_results1 = betas[:, 1]
    M_other = np.nanmean(betas[:, 2:], axis=1)

    t_taken = time() - t_st
    print(f'\tTime needed to regress: {t_taken:.2f} s')

    searched1 = results2img(eval_results1, second_level, mask, centers1,
                            mask_downsample_pre, resample1, downsample1)

    searched2 = results2img(M_other, second_level, mask, centers1,
                            mask_downsample_pre, resample1, downsample1)

    mask = ~np.isnan(searched1)

    return searched1, searched2, mask,




def eightlight(semantic=True, second_level='corr', flip=False,
               radius1=2, downsample1=1, resample1=1,
               radius2=2, downsample2=1, resample2=1,
               std=True, network=None,
               do_con=True, eightlight=True):

    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    # sns = sns[:20]

    title1, fn1, title2, fn2, title3, fn3 = get_titles_fns(
        flip, semantic, network, (60, 4 if do_con else 3), downsample1,
        radius1, downsample2, radius2, eightlight)
    fp3 = rf'result_pics/eightlight/{fn3}.png'
    if os.path.isfile(fp3):
        try:
            plot_searchlight_fn(fn1, dic='eightlight')
            plot_searchlight_fn(fn2, dic='eightlight')
            plot_searchlight_fn(fn3, dic='eightlight')
            return
        except FileNotFoundError as e:
            print(f'File not fond: {e}')

    fps = prep_fps('7')
    if not do_con: fps = [fp for fp in fps if 'con' not in fp]

    searched_all1 = []
    searched_all2 = []

    # t_last_plot = 0
    if network is not None:
        mask_ROIs = get_singular_ROIs(network)
    else:
        mask_ROIs = None

    base_shape = None
    # sns = sns[:30]
    # sns = sns[25:]
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
                  'second_level': second_level, 'flip': flip, 'std': std,
                  'mask_ROIs': mask_ROIs, }
            searched1, searched2, mask = utils.pickle_wrap(
                wrapped_jit_eightlight, kwargs=kw, verbose=-1,
                easy_override=False)
            if searched1 is None:
                assert base_shape is not None
                print(f'\tNONE NONE NONE: {sn}, {fp_fMRI_col}')
                searched1 = np.full(base_shape, np.nan)
                searched2 = np.full(base_shape, np.nan)
            else:
                assert base_shape is None or searched1.shape == base_shape
                if base_shape is None:
                    print(f'Base shape: {searched1.shape}')
                base_shape = searched1.shape

            t_needed = time() - t_st
            print(f'{sn}, {fp_fMRI_col}: '
                  f'{t_needed=:.2f}, {searched1.shape} | {kw=}')
            sn_l1.append(searched1)
            sn_l2.append(searched2)
        searched_all1.append(sn_l1)
        searched_all2.append(sn_l2)

    searched_all1_ = np.array(searched_all1)
    searched_all2_ = np.array(searched_all2)
    dif_all = searched_all2_ - searched_all1_
    sample_size = searched_all1_.shape[:2]

    searched_all1_ = np.nanmean(searched_all1_, axis=1)
    searched_all2_ = np.nanmean(searched_all2_, axis=1)
    dif_all_ = np.nanmean(dif_all, axis=1)
    t1 = get_t(searched_all1_)
    tile1 = np.nanquantile(t1, 0.995)
    t2 = get_t(searched_all2_)
    tile2 = np.nanquantile(t2, 0.995)
    tile_max = max(tile1, tile2)
    t_dif = get_t(dif_all_)
    tile_dif = np.nanquantile(np.abs(t_dif), 0.99)

    dif_min = np.nanmin(t_dif)
    print(f'{dif_min=:.2f}')
    dif_max = np.nanmax(t_dif)
    print(f'{dif_max=:.2f}')

    title1, fn1, title2, fn2, title3, fn3 = get_titles_fns(
        flip, semantic, network, sample_size, downsample1, radius1,
        downsample2, radius2, eightlight)

    plot_t(t1, title=title1, vabs=tile_max, fn=fn1, dic='eightlight')
    plot_t(t2, title=title2, vabs=tile_max, fn=fn2, dic='eightlight')
    plot_t(t_dif, title=title3, vabs=tile_dif, fn=fn3, only_positive=False,
           flip_color=True, dic='eightlight')

def get_titles_fns(flip, semantic, network, sample_size, downsample1, radius1,
               downsample2, radius2, eightlight=False):
    flip_str = ' flip' if flip else ''
    flip_str_ = '_flip' if flip else ''
    semantic_str = 'semantic' if semantic else 'visual'
    network_str = f'_{network}' if network is not None else ''

    title1 =  (f'Reg 1, {semantic_str}, N={sample_size}. '
               f'Down: {downsample1}, radius: {radius1}. '
               f'Control: ({downsample2}, {radius2})') + flip_str

    fn1 = (f'd{downsample1}-r{radius1}_d{downsample2}-r{radius2}{flip_str_}_'
           f'{semantic_str}_{sample_size}{network_str}_Reg1')

    fn2 = (f'd{downsample1}-r{radius1}_d{downsample2}-r{radius2}{flip_str_}_'
           f'{semantic_str}_{sample_size}{network_str}_Reg2')
    title2 =  (f'Reg 2, {semantic_str}, N={sample_size}. '
               f'Down: {downsample2}, radius: {radius2}. '
               f'Control: ({downsample1}, {radius1})') + flip_str

    fn3 = (f'd{downsample1}-r{radius1}_vs_d{downsample2}-r{radius2}{flip_str_}_'
           f'{semantic_str}_{sample_size}{network_str}_dif')
    title3 =  (f'Dif, {semantic_str}, N={sample_size}. '
               f'Blue: d={downsample1}, r={radius1}. '
               f'Red: d={downsample2}, r={radius2}.') + flip_str

    if eightlight:
        title1 = title1.replace('Reg 1', 'Multi')
        fn1 = fn1.replace('Reg1', 'Multi1')
        title2 = title2.replace('Reg 2', 'Multi')
        fn2 = fn2.replace('Reg2', 'Multi2')
        title3 = title3.replace('Dif', 'Dif Multi')
        fn3 = fn3.replace('dif', 'Multi_dif')

    return title1, fn1, title2, fn2, title3, fn3

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
    # TODO: calculate number of voxels contributing to each searchlight
    #   percentage filled by location...

    set_num_threads(1)
    # for NETWORK in ['OC_T', 'OC_IT']:
    # for NETWORK in ['OC_T', 'IT']:
    DO_CON = True
    for NETWORK in ['OC_IT', None, 'Occipital', 'IT']: # 'OC_T', None
        for SEMANTIC in [ True, False]: # False,
            # eightlight(semantic=SEMANTIC,
            #            radius1=3, downsample1=6, resample1=1,
            #            radius2=3, downsample2=3, resample2=1,
            #            network=NETWORK, do_con=DO_CON)
            #
            # eightlight(semantic=SEMANTIC,
            #            radius1=8, downsample1=3, resample1=1,
            #            radius2=4, downsample2=3, resample2=1,
            #            network=NETWORK, do_con=DO_CON)
            # quit()
            #
            # eightlight(semantic=SEMANTIC,
            #            radius1=3, downsample1=8, resample1=1,
            #            radius2=3, downsample2=4, resample2=1,
            #            network=NETWORK, do_con=DO_CON)

            # eightlight(semantic=SEMANTIC,
            #            radius1=3, downsample1=9, resample1=1,
            #            radius2=3, downsample2=3, resample2=1,
            #            network=NETWORK, do_con=DO_CON)
            #
            # eightlight(semantic=SEMANTIC,
            #            radius1=2, downsample1=10, resample1=1,
            #            radius2=4, downsample2=2, resample2=1,
            #            network=NETWORK, do_con=DO_CON)
            #
            eightlight(semantic=SEMANTIC,
                       radius1=1, downsample1=10, resample1=1,
                       radius2=3, downsample2=2, resample2=1,
                       network=NETWORK, do_con=DO_CON)
            # #
            # eightlight(semantic=SEMANTIC,
            #            radius1=2, downsample1=9, resample1=1,
            #            radius2=4, downsample2=3, resample2=1,
            #            network=NETWORK, do_con=DO_CON)
            #
            # eightlight(semantic=SEMANTIC,
            #            radius1=5, downsample1=4, resample1=1,
            #            radius2=5, downsample2=2, resample2=1,
            #            network=NETWORK, )

            # eightlight(semantic=SEMANTIC,
            #            radius1=4, downsample1=5, resample1=1,
            #            radius2=4, downsample2=2, resample2=1,
            #            network=NETWORK, )

            # eightlight(semantic=SEMANTIC,
            #            radius1=2, downsample1=8, resample1=1,
            #            radius2=6, downsample2=1, resample2=1,
            #            network=NETWORK, )
            # # quit()



