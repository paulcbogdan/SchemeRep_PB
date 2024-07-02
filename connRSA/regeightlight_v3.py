import os.path
from collections import defaultdict
from time import time
import numpy as np
from matplotlib import pyplot as plt

import utils

from connRSA.jit_funcs import evaluate_regresslight, get_closest_dists, evaluate_regresslight_std, \
    evaluate_multilight_std, evaluate_models_searchlight, jit_searchlight_orged_RDMs
from connRSA.searchlight_plot import plot_t
from connRSA.searchlight_v2 import full_get_RDMs, results2img, get_singular_ROIs, get_RSM_stim_flat, \
    plot_searchlight_fn, make_tril_mask_within_nan
from connRSA.conn_utils import mask_img
from connRSA.single_trial_conn import prep_fps
from org_sns import get_sns
from numba import set_num_threads

def wrapped_jit_cubelight(sn, fp_fMRI_col, semantic,
                          radius1=2, downsample1=1, resample1=1,
                          radius2=2, downsample2=1, resample2=1,
                          second_level='spear', flip=False, std=True,
                          mask_ROIs=None, mult27=False, m_rsm=False,
                          threshold=0.25, super64=False):
    print()
    kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
           'flip': flip,}
    RSM_stim_flat = utils.pickle_wrap(get_RSM_stim_flat, kwargs=kw1,
                                      verbose=0, easy_override=False)

    if super64:
        radius1_ = radius1 * 2
    else:
        radius1_ = int(radius1 * 1.5 + .0001)
    kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col,
           'flip': flip, 'radius': radius1_, 'downsample': downsample1,
           'resample': resample1, 'mask_ROIs': mask_ROIs,
           'threshold': threshold, 'no_RDMs': True}
    centers1, mask, mask_downsample_pre = (
        utils.pickle_wrap(full_get_RDMs, kwargs=kw1, verbose=0,
                          easy_override=False))
    if centers1 is None:
        return [None] * 4

    kw2 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col,
           'flip': flip, 'radius': radius2, 'downsample': downsample2,
           'resample': resample2, 'mask_ROIs': mask_ROIs,
           'threshold': threshold, 'get_sphere_Ms': True}
    fMRI_RDMs2, _, centers2, _, data_Ms = utils.pickle_wrap(full_get_RDMs,
                                                   kwargs=kw2, verbose=0,
                                                   easy_override=False)
    if fMRI_RDMs2 is None:
        return [None] * 4

    print(f'{data_Ms.shape=}')
    mult1 = downsample1 / downsample2
    ar = get_indexing_ar(centers1, centers2, radius1, mult1, mult27,
                         super64=super64)
    cube1s = data_Ms[:, ar]
    fMRI_RDMs1 = jit_searchlight_orged_RDMs(cube1s)
    tril_mask, kept_in = make_tril_mask_within_nan(flip=flip)
    fMRI_RDMs1 = fMRI_RDMs1[:, kept_in]

    print(f'{fMRI_RDMs1.shape}')

    regressors = fMRI_RDMs2[ar]
    if m_rsm:
        regressors = np.nanmean(regressors, axis=1)[:, None, :]

    t_st = time()
    betas = evaluate_multilight_std(fMRI_RDMs1, regressors, RSM_stim_flat)

    eval_results1 = betas[:, 1]
    M_other = np.nanmean(betas[:, 2:], axis=1)
    M_sum_other = np.nansum(betas[:, 2:], axis=1)
    print(f'\tTime needed to regress: {time() - t_st:.2f} s')

    searched1 = results2img(eval_results1, second_level, mask, centers1,
                            mask_downsample_pre, resample1, downsample1)

    searched2 = results2img(M_other, second_level, mask, centers1,
                            mask_downsample_pre, resample1, downsample1)
    if m_rsm:
        searched3 = None
    else:
        searched3 = results2img(M_sum_other, second_level, mask, centers1,
                                mask_downsample_pre, resample1, downsample1)

    mask = ~np.isnan(searched1)

    return searched1, searched2, searched3, mask,




def get_indexing_ar(centers1, centers2, radius1, mult1, mult27=False,
                    super64=False):
    t_st = time()
    idx1to_center = get_closest_dists(centers1.astype(np.float32),
                                      centers2.astype(np.float32),
                                      mult1=mult1)
    print(f'\tTime needed to find closest: {time() - t_st:.2f} s')

    i2new_spot = defaultdict(list)
    mod2idxs = {}

    centers1_multd = (centers1 * mult1).astype(np.float32)

    if super64:
        off1 = int(mult1)
        off2 = int(mult1 * 3)
        l = [-off2, -off1, off1, off2]
    else:
        off = int(radius1 * mult1)
        l = [-off, 0, off]

    for x_mod in l:
        for y_mod in l:
            for z_mod in l:
                num_zeros = np.sum([x_mod == 0, y_mod == 0, z_mod == 0])

                if super64:
                    pass
                elif mult27:
                    if num_zeros == 3:
                        continue
                else:
                    if num_zeros != 2:
                        continue


                if super64:
                    tup = (x_mod, y_mod, z_mod)
                else:
                    tup = (np.sign(x_mod), np.sign(y_mod), np.sign(z_mod))

                mod2idxs[tup] = (
                    get_closest_dists(centers1_multd +
                                      [x_mod, y_mod, z_mod],
                                      centers2.astype(np.float32),
                                      mult1=1))

                for i in range(len(centers1)):
                    i2new_spot[i].append(
                        (centers1[i] + [x_mod, y_mod, z_mod]))

    ar = np.vstack([l for l in mod2idxs.values()]).T

    nan_out = []
    num2is = defaultdict(list)
    for i in range(ar.shape[0]):
        num_unique = len(np.unique(ar[i]))
        num2is[num_unique].append(i)
    M_num_unique = np.mean([len(l) for l in num2is.values()])
    print(f'Mean number of uniques: {M_num_unique:.2f} ({mult27=})')

    if not super64:
        ar = np.concatenate([idx1to_center[:, None], ar], axis=1)
    if super64:
        assert ar.shape[1] == 64
    return ar


def wrapped_jit_eightlight(sn, fp_fMRI_col, semantic,
                           radius1=2, downsample1=1, resample1=1,
                           radius2=2, downsample2=1, resample2=1,
                           second_level='spear', flip=False, std=True,
                           mask_ROIs=None, mult27=False, m_rsm=False,
                           threshold=0.25):
    print()
    kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
           'flip': flip,}
    RSM_stim_flat = utils.pickle_wrap(get_RSM_stim_flat, kwargs=kw1,
                                      verbose=0, easy_override=False)
    assert np.sum(np.isnan(RSM_stim_flat)) == 0, \
        f'{np.sum(np.isnan(RSM_stim_flat))=}'

    kw1 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col,
           'flip': flip, 'radius': radius1, 'downsample': downsample1,
           'resample': resample1, 'mask_ROIs': mask_ROIs,
           'threshold': threshold}
    fMRI_RDMs1, mask, centers1, mask_downsample_pre = (
        utils.pickle_wrap(full_get_RDMs, kwargs=kw1, verbose=0,
                          easy_override=False))
    # fMRI_RDMs1, mask, centers1, mask_downsample_pre = full_get_RDMs(**kw1)

    if fMRI_RDMs1 is None:
        return None, None, None, None

    kw2 = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col,
           'flip': flip, 'radius': radius2, 'downsample': downsample2,
           'resample': resample2, 'mask_ROIs': mask_ROIs,
           'threshold': threshold}

    fMRI_RDMs2, _, centers2, _ = utils.pickle_wrap(full_get_RDMs,
                                                   kwargs=kw2, verbose=0,
                                                   easy_override=False)

    # fMRI_RDMs2, _, centers2, _ = full_get_RDMs(**kw2)

    print(f'{centers1.shape=}')
    print(f'{centers2.shape=}')
    if fMRI_RDMs2 is None:
        return None, None, None, None

    prop_nan1 = np.sum(np.isnan(fMRI_RDMs1)) / fMRI_RDMs1.size
    prop_nan2 = np.sum(np.isnan(fMRI_RDMs2)) / fMRI_RDMs2.size
    assert prop_nan1 == 0, f'{prop_nan1=}'
    assert prop_nan2 == 0, f'{prop_nan2=}'

    mult1 = downsample1 / downsample2
    ar = get_indexing_ar(centers1, centers2, radius1, mult1, mult27)

    regressors = fMRI_RDMs2[ar]

    if m_rsm:
        regressors = np.nanmean(regressors, axis=1)[:, None, :]

    del fMRI_RDMs2
    assert regressors[:, 0, :].shape == fMRI_RDMs1.shape
    assert np.sum(np.isnan(regressors)) == 0, \
        f'{np.sum(np.isnan(regressors))=}'

    t_st = time()

    betas = evaluate_multilight_std(fMRI_RDMs1, regressors, RSM_stim_flat)
    eval_results1 = betas[:, 1]
    M_other = np.nanmean(betas[:, 2:], axis=1)
    M_sum_other = np.nansum(betas[:, 2:], axis=1)

    t_taken = time() - t_st
    print(f'\tTime needed to regress: {t_taken:.2f} s')

    searched1 = results2img(eval_results1, second_level, mask, centers1,
                            mask_downsample_pre, resample1, downsample1)

    searched2 = results2img(M_other, second_level, mask, centers1,
                            mask_downsample_pre, resample1, downsample1)
    if m_rsm:
        searched3 = None
    else:
        searched3 = results2img(M_sum_other, second_level, mask, centers1,
                                mask_downsample_pre, resample1, downsample1)

    mask = ~np.isnan(searched1)

    return searched1, searched2, searched3, mask,




def eightlight(semantic=True, second_level='corr', flip=False,
               radius1=2, downsample1=1, resample1=1,
               radius2=2, downsample2=1, resample2=1,
               std=True, network=None,
               do_con=True, eightlight=True, mult27=False,
               use_sum_reg=False, m_rsm=False,
               threshold=0.25, small_Ms=False, super64=False):

    assert not (m_rsm and use_sum_reg)
    age2sn = get_sns('all', sh=False)
    sns = age2sn[1] + age2sn[2]
    # sns = sns[:20]

    title1, fn1, title2, fn2, title3, fn3 = get_titles_fns(
        flip, semantic, network, (60, 4 if do_con else 3), downsample1,
        radius1, downsample2, radius2, eightlight, use_sum_reg, mult27,
        m_rsm, threshold, small_Ms, super64)
    dic = 'cubelight' if small_Ms else 'eightlight'
    fp3 = rf'result_pics/{dic}/{fn3}.png'
    if os.path.isfile(fp3):
        try:
            plot_searchlight_fn(fn1, dic=dic)
            plot_searchlight_fn(fn2, dic=dic)
            plot_searchlight_fn(fn3, dic=dic)
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
    # sns = sns[:8] + sns[-25:-18]
    # sns = sns[:10]
    for sn_i, sn in enumerate(sns):
        sn_l1 = []
        sn_l2 = []
        for fp_fMRI_col in fps:
            # if 'vis' in fp_fMRI_col: continue
            t_st = time()
            kw = {'sn': sn, 'fp_fMRI_col': fp_fMRI_col, 'semantic': semantic,
                  'radius1': radius1, 'downsample1': downsample1,
                  'resample1': resample1,
                  'radius2': radius2, 'downsample2': downsample2,
                  'resample2': resample2,
                  'second_level': second_level, 'flip': flip, 'std': std,
                  'mask_ROIs': mask_ROIs, 'mult27': mult27,
                  'm_rsm': m_rsm, 'threshold': threshold,}
            if small_Ms:
                kw['super64'] = super64
                searched1, searched2, searched3, mask = utils.pickle_wrap(
                    wrapped_jit_cubelight, kwargs=kw, verbose=-1,
                    easy_override=False)
            else:
                searched1, searched2, searched3, mask = utils.pickle_wrap(
                    wrapped_jit_eightlight, kwargs=kw, verbose=-1,
                    easy_override=False)
            if use_sum_reg:
                searched2 = searched3
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
        downsample2, radius2, eightlight, use_sum_reg, mult27, m_rsm,
        threshold, small_Ms, super64)

    plot_t(t1, title=title1, vabs=tile_max, fn=fn1, dic=dic)
    plot_t(t2, title=title2, vabs=tile_max, fn=fn2, dic=dic)
    plot_t(t_dif, title=title3, vabs=tile_dif, fn=fn3, only_positive=False,
           flip_color=True, dic=dic)

def get_titles_fns(flip, semantic, network, sample_size, downsample1, radius1,
                   downsample2, radius2, eightlight=False, use_sum_reg=False,
                   mult27=False, m_rsm=False, threshold=0.25, small_Ms=False,
                   super64=False):
    flip_str = ' flip' if flip else ''
    flip_str_ = '_flip' if flip else ''
    semantic_str = 'semantic' if semantic else 'visual'
    network_str = f'_{network}' if network is not None else ''

    title1 =  (f'Reg 1, {semantic_str}, N={sample_size}. '
               f'Down: {downsample1}, radius: {radius1}. '
               f'Control: ({downsample2}, {radius2})') + flip_str

    core = (f'd{downsample1}-r{radius1}_d{downsample2}-r{radius2}{flip_str_}_'
            f'{semantic_str}_{sample_size}{network_str}')

    title2 =  (f'Reg 2, {semantic_str}, N={sample_size}. '
               f'Down: {downsample2}, radius: {radius2}. '
               f'Control: ({downsample1}, {radius1})') + flip_str

    title3 =  (f'Dif, {semantic_str}, N={sample_size}. '
               f'Blue: d={downsample1}, r={radius1}. '
               f'Red: d={downsample2}, r={radius2}.') + flip_str

    if eightlight:
        title1 = title1.replace('Reg 1', 'Multi')
        # fn1 = fn1.replace('Reg1', 'Multi1')
        title2 = title2.replace('Reg 2', 'Multi')
        # fn2 = fn2.replace('Reg2', 'Multi2')
        title3 = title3.replace('Dif', 'Dif Multi')
        # fn3 = fn3.replace('dif', 'Multi_dif')


    if use_sum_reg:
        core += '_sum'

    if super64:
        core += '_super64'
    elif mult27:
        core += '_mult27'
    if m_rsm:
        core += '_M_rsm'
    if threshold < 0.24 or threshold > 0.26:
        core += f'_t{threshold}'
    if small_Ms:
        core += '_small_Ms'

    fn1 = core + '_Multi1'
    fn2 = core + '_Multi2'
    fn3 = core + '_Multi_dif'

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
    MULT27 = False
    USE_SUM_REG = False
    M_RSM = False
    SMALL_Ms = True
    THRESHOLD = 0.5
    SUPER64 = False
    for MULT27 in [True, False]:  # True, False,
        for SEMANTIC in [True, False]:  # False,
            for NETWORK in ['cortex', 'OC_IT', 'OC_T']:  #'OC_IT',   'OC_IT',  , False, 'Occipital' None, 'Occipital', 'IT']: # 'OC_T', None  'OC_IT',
                eightlight(semantic=SEMANTIC,
                           radius1=2, downsample1=6, resample1=1,
                           radius2=6, downsample2=1,
                           resample2=1,#10 if NETWORK == 'cortex' else 1,
                           network=NETWORK, do_con=DO_CON,
                           use_sum_reg=USE_SUM_REG, mult27=MULT27,
                           m_rsm=M_RSM, threshold=THRESHOLD,
                           small_Ms=SMALL_Ms, super64=False)

                # eightlight(semantic=SEMANTIC,
                #            radius1=2, downsample1=5, resample1=1,
                #            radius2=5, downsample2=1,
                #            resample2=10 if NETWORK == 'cortex' else 1,
                #            network=NETWORK, do_con=DO_CON,
                #            use_sum_reg=USE_SUM_REG, mult27=MULT27,
                #            m_rsm=M_RSM, threshold=THRESHOLD,
                #            small_Ms=SMALL_Ms, super64=False)
