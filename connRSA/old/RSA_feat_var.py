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
from numba import jit, prange

import os
# os.chdir(r'/')

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

def shuffle_d_vecs(d_vecs, seed=0):
    rng = np.random.default_rng(seed)
    d_vecs_new = {}
    keys = list(d_vecs.keys())
    vals = list(d_vecs.values())
    rng.shuffle(keys)

    for i, key in enumerate(keys):
        d_vecs_new[key] = vals[i]
    return d_vecs_new

@cache
def prep_w2v_feature_RSMs(sn, fp, semantic, flat=True,
                          shuffle_seed=None):
    d_vecs = prep_vecs(True, semantic)
    if shuffle_seed is not None:
        d_vecs = shuffle_d_vecs(d_vecs, seed=shuffle_seed)

    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').
            replace('4', '').replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)  # added to help with run-wise sorting
    vecs_all = np.array([d_vecs[obj] for obj in df_sn['obj']]).T
    vecs_all = stats.zscore(vecs_all, axis=1)

    RSMs = np.abs(vecs_all[:, :, None] - vecs_all[:, None, :])

    RSMs = np.array(RSMs)
    if flat:
        tril_idxs = np.tril_indices(114, k=-1)
        RSM_models_flat = RSMs[:, *tril_idxs]
        RSM_models_flat = stats.zscore(RSM_models_flat, axis=1,
                                       nan_policy='omit')
        RSM_models_flat = RSM_models_flat.astype(np.float32)
        return RSM_models_flat
    else:
        return RSMs

def analyze_var_ROI(sn, ROI, fp, trial_similarity, stdize_by_run,
                    second_order, semantic, regress_global=True,
                    ROI1=None, RSM1=None, shuffle_seed=None):
    tril_idxs = np.tril_indices(114, k=-1)
    if RSM1 is not None:
        RSM_fMRI_flat = RSM1
    else:
        # if ROI1 is not None:
        #     dir_in = fr'cache/conn_RSA/ars/RSA_reg'
        #     Path(dir_in).mkdir(parents=True, exist_ok=True)
        #     fp_focus1 = (f'{dir_in}/{sn}_{ROI}_{ROI1}_{trial_similarity}_'
        #                  f'{second_order}_{stdize_by_run}_BOLD.npy')
        # else:
        #     dir_in = fr'cache/conn_RSA/ars/RSA'
        #     RDM_method_ = 'within_nan'
        #     dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
        #                  f'{second_order}_{RDM_method_}_{stdize_by_run}')
        #     fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
        # with open(fp_focus1, 'rb') as f:
        #     RSM_focus1 = np.load(f)
        # RSM_focus1 = within_run_to_nan(RSM_focus1)

        RSM_fMRI_flat, status = grab_ROI_RSM(fp, trial_similarity,
                                     second_order, stdize_by_run,
                     sn, ROI, within_nan=True, flat=True, z=True)



        # RSM_fMRI_flat = RSM_focus1[tril_idxs]

    kw_ = {'sn': sn, 'fp': fp, 'semantic': semantic, 'flat': True,
           'shuffle_seed': shuffle_seed}
    RSM_models_flat = pickle_wrap(prep_w2v_feature_RSMs, kwargs=kw_,
                                  verbose=-1)

    # RSM_models_flat = prep_w2v_feature_RSMs(sn, fp, semantic, flat=True,
    #                                         shuffle_seed=shuffle_seed)


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
        t_st = time()
        RSM_corr = np.nanmean(RSM_fMRI_flat[None,  :] *
                              RSM_models_flat, axis=1)
        t_end = time()
        t_dif = t_end - t_st
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

@cache
def get_regressed_w2v_RSMs(sn, fp, semantic, trial_similarity,
                           second_order, stdize_by_run):
    # takes ~ 4 seconds
    ROIs = get_atlas()['ROIs']
    RSMs = prep_w2v_feature_RSMs(sn, fp, semantic, flat=True)
    RSMs_l = []
    for ROI in tqdm(ROIs):
        RSM_focus1, status = grab_ROI_RSM(fp, trial_similarity, second_order,
                                          stdize_by_run, sn, ROI,
                                          within_nan=True, flat=True)
        if not status:
            RSMs_l.append(np.full(RSMs.shape, np.nan))
            continue

        # print()
        regressors = np.vstack([RSM_focus1, np.ones(len(RSM_focus1))]).T
        nans = np.any(np.isnan(regressors), axis=1)
        regressors = regressors[~nans, :]

        XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
        XTX_invX = np.dot(XTX_inv, regressors.T)
        betas = np.dot(XTX_invX, RSMs[:, ~nans].T)
        betas0 = betas[0, :]

        RSMs_ = RSMs - betas0[:, None] * RSM_focus1[None, :]
        RSMs_l.append(RSMs_)

        # RSMs1.append(RSM_focus1)
    RSMs_l = np.array(RSMs_l)
    # quit()
    return RSMs_l
    # print(RSMs_l.shape)
    # quit()
    # RSMs1, status = regress_out_RSMs(sn, None, fp, trial_similarity,
    #                                  stdize_by_run, second_order, override=True)

def get_sn_regressed_corr_fp(sn, fp, trial_similarity, stdize_by_run,
                             semantic, second_order, ROIs,
                             full_regress_out=True):#
    do_nothing = False
    ars = []
    RSMs1_all = []
    tril_idxs = np.tril_indices(114, k=-1)
    if full_regress_out:
        kw = {'sn': sn, 'fp': fp, 'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run, 'semantic': semantic,
              'second_order': second_order}
        RSM_models_flat = pickle_wrap(get_regressed_w2v_RSMs, kwargs=kw,)
        # print(f'{RSM_models_flat.shape=}')
        # quit()
    else:
        RSM_models_flat = prep_w2v_feature_RSMs(sn, fp, semantic)


    for ROI0 in tqdm(ROIs, desc=f'Looping local regressed ROIs for: {sn}'):
        RSMs1, status = regress_out_RSMs(sn, ROI0, fp, trial_similarity,
                                         stdize_by_run, second_order,
                                         override=True,
                                         do_nothing=do_nothing)

        if status:
            RSMs1_all.append(RSMs1)
        elif not status:
            RSMs1_all.append(np.full((246, 6441), np.nan))

    RSMs1_all = np.array(RSMs1_all, dtype=np.float32)


    RSMs1_all = stats.zscore(RSMs1_all, axis=-1, nan_policy='omit')
    t_st = time()

    nan_cells = np.all(np.isnan(RSMs1_all), axis=(0, 1))


    RSM_models_flat = RSM_models_flat[..., ~nan_cells]
    RSMs1_all = RSMs1_all[:, :, ~nan_cells]
    if len(RSM_models_flat.shape) == 3:
        # print('SUPE')
        # quit()
        ars = numba_super_alt(RSMs1_all, RSM_models_flat) # actually like 80% more time
    else:
        ars = numba_super(RSMs1_all, RSM_models_flat)

    t_end = time()
    t_dif = t_end - t_st
    print(f'Super numba time: {t_dif=:.3f}')
    # print(f'{ars.shape=}')
    # quit()
    # (N_ROI_regressed, N_ROI, N_dims)
    return ars


def get_sn_regressed_corr(sn, fps, trial_similarity, stdize_by_run,
                          semantic, second_order, full_regress_out=True,
                          ): # do_nothing=False
    atlas = get_atlas()
    ROIs = atlas['ROIs']

    # fps = ['obj7_fMRI']
    ars = []
    for fp in fps:
        kw = {'sn': sn, 'fp': fp, 'trial_similarity': trial_similarity,
                'stdize_by_run': stdize_by_run, 'semantic': semantic,
                'second_order': second_order, 'ROIs': ROIs,
              'full_regress_out': full_regress_out}#, 'do_nothing': do_nothing}

        ar = pickle_wrap(get_sn_regressed_corr_fp, kwargs=kw, verbose=0,
                         easy_override=False)
        # print(kw['full_regress_out'])
        # quit()
        # print(ar)

        # kw = {'sn': sn, 'fp': fp, 'trial_similarity': trial_similarity,
        #       'stdize_by_run': stdize_by_run, 'semantic': semantic,
        #       'second_order': second_order, 'ROIs': ROIs,
        #       'full_regress_out': False}
        # ar = pickle_wrap(get_sn_regressed_corr_fp, kwargs=kw, verbose=0,
        #                  easy_override=False)
        # print(ar.shape)
        # quit()
        ars.append(ar)
    ars = np.array(ars)

    return ars

def get_ars_standard(sns, semantic, kwargs):
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
    return sn_all_vals_l

def get_corr_standard(sns, four_tasks, trial_similarity, stdize_by_run,
                      semantic, second_order, regress_global=False):
    # isn't just ROI-specific lines like the local one
    atlas = get_atlas()
    ROIs = atlas['ROIs']
    sns = sns[:3]

    kwargs = {'sns': sns, 'four_tasks': four_tasks,
              'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'semantic': semantic, 'second_order': second_order,
              'regress_global': regress_global}

    sn_all_vals_l = get_ars_standard(sns, semantic, kwargs)
    sn_all_vals = sn_all_vals_l[:, :, 0] # only bl7
    sn_all_vals = np.transpose(sn_all_vals, (1, 0, 2))
    sn_all_vals_conn = np.nanmean((sn_all_vals[..., None, :] *
                                   sn_all_vals[..., None, :, :]), axis=-1)

    conn_l = sn_all_vals_conn
    return conn_l

def get_sn_no_regresssed_corr(sn, fps, trial_similarity, stdize_by_run,
                              semantic, second_order, regress_global=False):
    kwargs = {
              'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'semantic': semantic, 'second_order': second_order,
              'regress_global': regress_global}
    ar = []
    for fp in fps:
        l = []
        for ROI in get_atlas()['ROIs']:
            kwargs['ROI'] = ROI
            kwargs['fp'] = fp

            try:
                vals = analyze_var_ROI(sn, ROI, fp, trial_similarity, stdize_by_run,
                        second_order, semantic, regress_global=False)
            except FileNotFoundError as e:
                l.append(np.full(300 if semantic else 114, np.nan))
                continue

            l.append(vals)
        ar.append(l)
    return np.array(ar)


def get_sn_RSA_feat_conn(sn, four_tasks, trial_similarity,
                         stdize_by_run, semantic, second_order,
                         full_regress_out=True, ):

    fps = prep_fps(four_tasks)
    fps = ['bl7_fMRI']
    kwargs = {'sn': sn, 'fps': fps,
              'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'semantic': semantic, 'second_order': second_order,
              'regress_global': False}
    ars_no = pickle_wrap(get_sn_no_regresssed_corr, kwargs=kwargs, verbose=0)

    # ars_no = get_sn_no_regresssed_corr(sn , fps, trial_similarity,
    #                                    stdize_by_run, semantic, second_order)

    ars = get_sn_regressed_corr(sn, fps, trial_similarity,
                                stdize_by_run, semantic, second_order,
                                full_regress_out=full_regress_out,
                                ) # do_nothing=do_nothing


    sn_l = []
    for fp_k in range(ars.shape[0]):
        conn = np.full((246, 246), np.nan)
        for i in range(246):
            for j in range(246):
                conn[i, j] = np.mean(ars[fp_k, j, i, :] *
                                     ars_no[fp_k, j, :])
                # conn[i, j] = np.mean(ars[fp_k, j, i, :] *
                #                      ars[fp_k, i, j, :] )
        conn[np.diag_indices_from(conn)] = np.nan
        plt.title(f'{sn}, {fp_k}, {semantic=}')
        plt.imshow(conn)
        plt.show()
        # quit()
        sn_l.append(conn)
    return np.nanmean(sn_l, axis=0)

def get_corr_regress_local(sns, four_tasks, trial_similarity, stdize_by_run,
                           semantic, second_order, full_regress_out=True):
    # l = []
    conn_l = []
    for i, sn in enumerate(sns):

        kw = {'sn': sn, 'four_tasks': four_tasks,
              'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'semantic': semantic, 'second_order': second_order,
              'full_regress_out': full_regress_out}
        conn = pickle_wrap(get_sn_RSA_feat_conn, kwargs=kw, verbose=-1,
                           easy_override=False)
        conn_l.append(conn)

    return conn_l

def run_var_analysis():
    semantic = True
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    second_order = 'spear'
    four_tasks = '8'
    regress_local = True
    regress_global = False
    full_regress_out = True

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117', '118', '119', '120',
           '123', '124', '126', '127', '128', '129', '130', '134', '135',
           '136', '137', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214', '216', '217', '218',
           '219', '221', '222', '225', '227', '232', '233', '235']
    # sns = sns[:3]
    sns = sns[:20]


    if regress_local:
        conn_l = get_corr_regress_local(sns, four_tasks, trial_similarity,
                                        stdize_by_run, semantic, second_order,
                                        full_regress_out=full_regress_out)
    else:
        conn_l = get_corr_standard(sns, four_tasks, trial_similarity,
                                   stdize_by_run, semantic, second_order,
                                   regress_global=regress_global)

    M = np.nanmean(conn_l, axis=0)
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


@jit(nopython=True, parallel=True, fastmath=True)
def numba_super(a, b):
    num_ROI = a.shape[0]
    num_ROI2 = a.shape[1]
    num_dims = b.shape[0]
    ars = np.empty((num_ROI, num_ROI2, num_dims))
    for i in prange(num_ROI):
        for j in range(num_ROI2):
            x = a[i, j]
            for k in range(num_dims):
                ars[i, j, k] = np.mean(x * b[k])
    return ars

@jit(nopython=True, parallel=True, fastmath=True)
def numba_super_alt(a, b):
    num_ROI = a.shape[0]
    num_ROI2 = a.shape[1]
    num_dims = b.shape[1]
    ars = np.empty((num_ROI, num_ROI2, num_dims))
    for i in prange(num_ROI): # regressed
        for j in range(num_ROI2): # actual of interest
            x = a[i, j]
            for k in range(num_dims):
                ars[i, j, k] = np.mean(x * b[i, k])
    return ars


def numpy_super(a, b):
    l = []
    b = b[None, :, :]
    for a_ in tqdm(a, desc='numpy test'):
        a_ = a_[:, None, :]
        ar = np.nanmean(a_ * b, axis=-1)
        l.append(ar)
    ars = np.array(l)
    return ars

#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
# def do_RSM_numba(RSMs1, RSM_flat0, regressors, nans):
#     XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
#     XTX_invX = np.dot(XTX_inv, regressors.T)
#     betas = np.dot(XTX_invX, RSMs1[:, ~nans].T)
#     betas0 = betas[0, :]
#     # print(RSM_flat0[None, :].shape)
#     # print(betas0[:, None].shape)
#     RSMs1 -= RSM_flat0[None, :] * betas0[:, None]
#
#     return RSMs1

@cache
def grab_ROI_RSM(fp, trial_similarity, second_order, stdize_by_run,
                 sn, ROI, within_nan=True, flat=True, z=False):
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
    if z: RSM_focus1 = stats.zscore(RSM_focus1, nan_policy='omit')

    return RSM_focus1, True


def regress_out_RSMs(sn, ROI0, fp, trial_similarity, stdize_by_run,
                     second_order, override=False, RSM_focus0=False,
                     do_nothing=False):
    atlas = get_atlas()

    # if RSM_focus0:
    #     pass
    # else:
    RSM_focus0, status = grab_ROI_RSM(fp, trial_similarity, second_order,
                                  stdize_by_run, sn, ROI0, within_nan=True,
                                  flat=True)
    if not status:
        return None, False
    # print(RSM_focus0.shape)
    # print(RSM_focus0)
    RSM_focus0 = stats.zscore(RSM_focus0, nan_policy='omit')

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
    if do_nothing:
        return RSMs1, True

    RSMs1 = stats.zscore(RSMs1, axis=1, nan_policy='omit')

    # print(RSMs1.shape)
    # quit()
    ones = np.ones(RSM_focus0.shape)
    regressors = np.array([RSM_focus0, ones]).T
    nans = np.any(np.isnan(regressors), axis=1)
    regressors = regressors[~nans, :]
    # print(RSMs1.dtype)
    # quit()


    XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
    XTX_invX = np.dot(XTX_inv, regressors.T)
    betas = np.dot(XTX_invX, RSMs1[:, ~nans].T)
    betas0 = betas[0, :]

    RSMs1 -= RSM_focus0[None, :] * betas0[:, None]

    return RSMs1, status


if __name__ == '__main__':


    run_var_analysis()


