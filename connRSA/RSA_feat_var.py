from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
from scipy import stats
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from old.plot_gen import plot_connectivity
from org_sns import get_sns
from organize_bhv import get_trial_info
from stim import get_semantic_vectors, get_stim_RDM
from utils import pickle_wrap
from functools import cache
from sklearn import decomposition
from time import time

import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')

@cache
def get_M_RSM(sn, fp, trial_similarity, stdize_by_run, second_order):
    atlas = get_atlas()
    ROIs = atlas['ROIs']
    RSMs_l = []
    for ROI in ROIs:
        dir_in = fr'cache/conn_RSA/ars/RSA'
        RDM_method_ = 'within_nan'
        dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
                     f'{second_order}_{RDM_method_}_{stdize_by_run}')
        fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
        if not os.path.isfile(fp_focus1):
            continue
        with open(fp_focus1, 'rb') as f:
            RSM_focus1 = np.load(f)
        RSMs_l.append(RSM_focus1)
    RSMs = np.array(RSMs_l)
    RSM = np.nanmean(RSMs, axis=0)
    return RSM

@cache
def prep_w2v_feature_RSMs(sn, fp, semantic):
    d_vecs = prep_vecs(True, semantic)

    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').
            replace('4', '').replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)  # added to help with run-wise sorting
    vecs_all = np.array([d_vecs[obj] for obj in df_sn['obj']]).T
    vecs_all = stats.zscore(vecs_all, axis=1)

    RSMs = vecs_all[:, :, None] - vecs_all[:, None, :]

    RSMs = np.array(RSMs)
    return RSMs

def analyze_var_ROI(sn, ROI, fp, trial_similarity, stdize_by_run,
                    second_order, semantic, regress_global=True,
                    ROI1=None, RSM1=None):
    tril_idxs = np.tril_indices(114, k=-1)
    if RSM1 is not None:
        RSM_fMRI_flat = RSM1
    else:
        if ROI1 is not None:
            dir_in = fr'cache/conn_RSA/ars/RSA_reg'
            Path(dir_in).mkdir(parents=True, exist_ok=True)
            fp_focus1 = (f'{dir_in}/{sn}_{ROI}_{ROI1}_{trial_similarity}_'
                         f'{second_order}_{stdize_by_run}_BOLD.npy')
        else:
            dir_in = fr'cache/conn_RSA/ars/RSA'
            RDM_method_ = 'within_nan'
            dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
                         f'{second_order}_{RDM_method_}_{stdize_by_run}')
            fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
        with open(fp_focus1, 'rb') as f:
            RSM_focus1 = np.load(f)
        RSM_focus1 = within_run_to_nan(RSM_focus1)


        RSM_fMRI_flat = RSM_focus1[tril_idxs]
        RSM_fMRI_flat = stats.zscore(RSM_fMRI_flat, nan_policy='omit')

    RSMs = prep_w2v_feature_RSMs(sn, fp, semantic)
    RSM_models_flat = RSMs[:, *tril_idxs]
    RSM_models_flat = stats.zscore(RSM_models_flat, axis=1)

    if regress_global:
        kw = {'sn': sn, 'fp': fp, 'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run, 'second_order': second_order}
        RSM_global = pickle_wrap(get_M_RSM, kwargs=kw, verbose=-1)
        RSM_global_flat = RSM_global[tril_idxs]
        ones = np.ones(RSM_fMRI_flat.shape)
        regressors = np.array([RSM_fMRI_flat, RSM_global_flat, ones]).T
        nans = np.any(np.isnan(regressors), axis=1)
        regressors = regressors[~nans, :]
        RSM_models_flat = RSM_models_flat[:, ~nans]
        XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
        XTX_invX = np.dot(XTX_inv, regressors.T)
        betas = np.dot(XTX_invX, RSM_models_flat.T)
        RSM_corr = betas[0, :]

    else:
        # t_st = time()
        RSM_corr = np.nanmean(RSM_fMRI_flat[None,  :] *
                              RSM_models_flat, axis=1)
        # t_end = time()
        # t_dif = t_end - t_st
        # print(f'{t_dif=:.3f} s')
        # quit()
    return RSM_corr

def analyze_var_sns(sns, ROI, four_tasks, trial_similarity, stdize_by_run,
                    semantic, second_order, regress_global=True):
    sn_all_vals = []
    fps = prep_fps(four_tasks)
    sns_used = []
    for sn in tqdm(sns):
        sn_vals = []
        for fp in fps:
            try:
                vals = analyze_var_ROI(sn, ROI, fp, trial_similarity,
                                       stdize_by_run, second_order, semantic,
                                       regress_global=regress_global)

            except FileNotFoundError as e:
                print(f'Missing ({sn}, {fp}): {e}')
                break
            sn_vals.append(vals)
        else:
            sns_used.append(sn)
            sn_all_vals.append(sn_vals)
    sn_all_vals = np.array(sn_all_vals)
    return sn_all_vals

def get_sn_regressed_corr_fp(sn, fp, trial_similarity, stdize_by_run,
                             semantic, second_order, ROIs):
    ars = []
    for ROI0 in tqdm(ROIs, desc=f'Looping local regressed ROIs for: {sn}'):
        RSMs1, status = regress_out_RSMs(sn, ROI0, fp, trial_similarity,
                               stdize_by_run, second_order, override=False)
        if not status:
            ars.append(np.full((len(ROIs), 300 if semantic else 114),
                              np.nan))
            continue

        tril_idxs = np.tril_indices(114, k=-1)
        RSMs_models = prep_w2v_feature_RSMs(sn, fp, semantic)
        RSM_models_flat = RSMs_models[:, *tril_idxs]
        RSM_models_flat = stats.zscore(RSM_models_flat, axis=1,
                                       nan_policy='omit')
        RSMs1 = stats.zscore(RSMs1, axis=-1, nan_policy='omit')
        # print(RSMs1.shape)
        # print(RSM_models_flat.shape)
        # t_st = time()
        ar = np.nanmean(RSMs1[:, None, :] * RSM_models_flat[None], axis=-1)
        # ar = numba_corr(RSMs1, RSM_models_flat)
        print(ar.shape)
        ars.append(ar)
        # t_end = time()
        # t_dif = t_end - t_st
        # print(f'{t_dif=:.3f}')
        # quit()
        # print(ar.shape)

        # quit()

        # l = []
        # for RSM1 in RSMs1:
        # # for ROI1 in ROIs:
        #     RSM0_corrs = analyze_var_ROI(sn, ROI0, fp, trial_similarity,
        #                                  stdize_by_run, second_order,
        #                                  semantic, regress_global=False,
        #                                  ROI1=None, RSM1=RSM1)
        #     l.append(RSM0_corrs)
        #     # print()
        # ar.append(l)
        # print(np.array(l).shape)
        # quit()
    return ar

def numba_mat_prod(a, b):

    pass

def get_sn_regressed_corr(sn, four_tasks, trial_similarity, stdize_by_run,
                          semantic, second_order):
    atlas = get_atlas()
    ROIs = atlas['ROIs']
    fps = prep_fps(four_tasks)
    ars = []
    for fp in fps:
        kw = {'sn': sn, 'fp': fp, 'trial_similarity': trial_similarity,
                'stdize_by_run': stdize_by_run, 'semantic': semantic,
                'second_order': second_order, 'ROIs': ROIs}
        ar = pickle_wrap(get_sn_regressed_corr_fp, kwargs=kw, verbose=0)
        ars.append(ar)
    ars = np.array(ars)
    return ars

def get_corr_standard(sns, four_tasks, trial_similarity, stdize_by_run,
                      semantic, second_order, regress_global=False):
    atlas = get_atlas()
    ROIs = atlas['ROIs']
    kwargs = {'sns': sns, 'four_tasks': four_tasks,
              'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'semantic': semantic, 'second_order': second_order,
              'regress_global': regress_global}
    sn_all_vals_l = []
    for i, ROI in enumerate(ROIs):
        kwargs['ROI'] = ROI
        sn_all_vals = pickle_wrap(analyze_var_sns, kwargs=kwargs,
                                  easy_override=False, verbose=-1)

        if sn_all_vals.shape[0] < len(sns):
            sn_all_vals_l.append(np.full((len(sns), 4,
                                          300 if semantic else 114), np.nan))
        else:
            sn_all_vals_l.append(sn_all_vals)
    sn_all_vals_l = np.array(sn_all_vals_l)
    sn_all_vals = sn_all_vals_l[:, :, 0]
    sn_all_vals = np.transpose(sn_all_vals, (1, 0, 2))
    sn_all_vals_conn = np.nanmean((sn_all_vals[..., None, :] *
                                   sn_all_vals[..., None, :, :]), axis=-1)
    conn_l = sn_all_vals_conn
    return conn_l

def get_corr_regress_local(sns, four_tasks, trial_similarity, stdize_by_run,
                           semantic, second_order):
    for sn in sns:
        print(f'sn: {sn}')
        ars = get_sn_regressed_corr(sn, four_tasks, trial_similarity,
                                    stdize_by_run, semantic, second_order)

def run_var_analysis():
    semantic = True
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    second_order = 'spear'
    four_tasks = '7'
    regress_local = True
    regress_global = False

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117', '118', '119', '120',
           '123', '124', '126', '127', '128', '129', '130', '134', '135',
           '136', '137', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214', '216', '217', '218',
           '219', '221', '222', '225', '227', '232', '233', '235']
    sns = sns[3::4]

    if regress_local:
        conn_l = get_corr_regress_local(sns, four_tasks, trial_similarity,
                                        stdize_by_run, semantic, second_order)
    else:
        conn_l = get_corr_standard(sns, four_tasks, trial_similarity,
                                   stdize_by_run, semantic, second_order,
                                   regress_global=regress_global)

    M = np.nanmean(conn_l, axis=0)
    print(M)
    SE = stats.sem(conn_l, axis=0, nan_policy='omit')
    N = np.sum(~np.isnan(conn_l), axis=0)
    t = M / SE
    # plt.imshow(t)
    # plt.show()
    # quit()
    # print(t)
    # z = stats.norm.ppf(stats.t.cdf(t, N - 1))
    # print(z)
    atlas = get_atlas()
    plot_connectivity(M, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title='eh', no_avg=True, cbar_label='t-value',)

from numba import jit

#@jit()
@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def numba_corr(a, b):
    ar = np.empty((a.shape[0], b.shape[0]))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            ar[i, j] = np.nanmean(x * y)
    return ar
    # RSM_flat0_ = repeatnumba(a, 246).T
    # print(RSM_flat0_.shape)
    # quit()
    # RSM_flat0_ = np.expand_dims(RSM_flat0, 0)
    # print(RSM_flat0_.shape)
    # RSM_flat0_ = np.repeat(RSM_flat0_, 246)
    # print(RSM_flat0_.shape)
    # quit()
    # print(betas0.shape)
    # betas0_ = repeatnumba(betas0, 6441)
    # print(betas0_.shape)
    # quit()


def repeatnumba(original,no_repeats):
  repeat=original.repeat(no_repeats).reshape(*original.shape, no_repeats )
  return repeat

#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def do_RSM_numba(RSMs1, RSM_flat0, regressors, nans):
    XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
    XTX_invX = np.dot(XTX_inv, regressors.T)
    betas = np.dot(XTX_invX, RSMs1[:, ~nans].T)
    betas0 = betas[0, :]
    # print(RSM_flat0[None, :].shape)
    # print(betas0[:, None].shape)
    RSMs1 -= RSM_flat0[None, :] * betas0[:, None]
    # quit()
    # RSM_flat0_ = repeatnumba(RSM_flat0, 246).T
    # print(RSM_flat0_.shape)
    # quit()
    # RSM_flat0_ = np.expand_dims(RSM_flat0, 0)
    # print(RSM_flat0_.shape)
    # RSM_flat0_ = np.repeat(RSM_flat0_, 246)
    # print(RSM_flat0_.shape)
    # quit()
    # print(betas0.shape)
    # betas0_ = repeatnumba(betas0, 6441)
    # print(betas0_.shape)
    # quit()


    # betas0_ = np.expand_dims(betas0, 1)
    # # print(f'{betas0.shape=}')
    # betas0_ = np.repeat(betas0_, 6441, 1)
    # print(betas0_.shape)
    # quit()
    # print(f'{RSM_flat0_.shape=}')
    # print(f'{betas0_.shape=}')
    # quit()
    # RSMs1 -= RSM_flat0_ * betas0_

    return RSMs1

@cache
def grab_ROI_RSM(fp, trial_similarity, second_order, stdize_by_run,
                 sn, ROI, within_nan=True, flat=True):
    dir_in = fr'cache/conn_RSA/ars/RSA'
    RDM_method_ = 'within_nan'
    dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
                  f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
    try:
        with open(fp_focus1, 'rb') as f:
            RSM_focus1 = np.load(f)
    except FileNotFoundError:
        return None, False
    if within_nan: RSM_focus1 = within_run_to_nan(RSM_focus1)
    if flat: RSM_focus1 = RSM_focus1[np.tril_indices(RSM_focus1.shape[0], k=-1)]
    return RSM_focus1, True


def regress_out_RSMs(sn, ROI0, fp, trial_similarity, stdize_by_run,
                     second_order, override=False):
    atlas = get_atlas()
    t_st = time()

    # last_ROI = atlas['ROIs'][-1]
    # dir_out = fr'cache/conn_RSA/ars/RSA_reg'
    # Path(dir_out).mkdir(parents=True, exist_ok=True)
    # last_out = (f'{dir_out}/{sn}_{ROI0}_{last_ROI}_{trial_similarity}_'
    #             f'{second_order}_{stdize_by_run}_BOLD.npy')
    # if os.path.isfile(last_out) and not override:
    #     return None, False

    RSM_focus0, status = grab_ROI_RSM(fp, trial_similarity, second_order,
                                      stdize_by_run, sn, ROI0, within_nan=True,
                                      flat=True)
    if not status:
        return None, False


    RSMs1 = []
    for ROI1 in atlas['ROIs']:
        RSM_focus2, status = grab_ROI_RSM(fp, trial_similarity, second_order,
                                          stdize_by_run, sn, ROI1,
                                          within_nan=True, flat=True)
        if not status:
            RSMs1.append(np.full(RSM_focus0.shape, np.nan))
            continue
        RSMs1.append(RSM_focus2)

    RSMs1 = np.array(RSMs1)
    ones = np.ones(RSM_focus0.shape)
    regressors = np.array([RSM_focus0, ones]).T
    nans = np.any(np.isnan(regressors), axis=1)
    regressors = regressors[~nans, :]

    RSMs1 = do_RSM_numba(RSMs1, RSM_focus0, regressors, nans)
    t_end = time()
    t_dif = t_end - t_st
    print(f'{t_dif=:.3f} s')
    return RSMs1, status

    # tril = np.tril_indices(114, -1)
    # for i, ROI1 in enumerate(atlas['ROIs']):
    #     fp_out = (f'{dir_out}/{sn}_{ROI0}_{ROI1}_{trial_similarity}_'
    #               f'{second_order}_{stdize_by_run}_BOLD.npy')
    #     RSM = np.full((114, 114), np.nan)
    #     RSM[tril] = RSMs1[i]
    #     RSM[tril[1], tril[0]] = RSMs1[i]
    #     with open(fp_out, 'wb') as f:
    #         np.save(f, RSM)

    # return True

if __name__ == '__main__':
    # for _ in range(100):
    #     regress_out_RSMs('102', get_atlas()['ROIs'][0], 'obj7_fMRI',
    #                      'corr', False, 'spear', override=True)
    #
    # regress_out_RSMs('102', get_atlas()['ROIs'][0], 'obj7_fMRI',
    #                  'corr', False, 'spear', override=True)
    # quit()


    run_var_analysis()


