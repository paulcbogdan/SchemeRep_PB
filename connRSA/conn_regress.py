import os

import pandas as pd
from tqdm import tqdm

from connRSA.conn_analyze_IRAFs import prep_network2ROI, ROI2NETWORK
from connRSA.conn_utils import get_BNA_ROIs
from connRSA.single_trial_conn import prep_fps
from fMRI_proc import within_run_to_nan
from old.networks import prep_networks
from utils import pickle_wrap

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

import numpy as np
import scipy.stats as stats

from time import time
from org_sns import get_sns
import matplotlib.pyplot as plt

def do_regr_ERS_sn(sn, ROI_focus_conn, ROIs_ctrl,
                   fp0, fp1, trial_similarity, stdize_by_run,
                   semantic, second_order, RDM_method, fp):
    dir_focus = fr'cache/conn_RSA/ars/ERS'
    dir_focus = fr'{dir_focus}/{fp0}_{fp1}_{trial_similarity}_{stdize_by_run}'
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus_conn}.npy'
    with open(fp_focus, 'rb') as f:
        ERS_focus = np.load(f)
    flat_focus = ERS_focus.flatten()

    if len(ROIs_ctrl):
        ERS_ctrl_l = []
        for ROI_ctrl in ROIs_ctrl:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            with open(fp_ctrl, 'rb') as f:
                ERS_ctrl = np.load(f)
            ERS_ctrl_l.append(ERS_ctrl)
        ERS_ctrl = np.array(ERS_ctrl_l)
        flat_ctrl = ERS_ctrl.reshape(ERS_ctrl.shape[0], -1).T # (12996, 22)
        nan_cols = np.isnan(flat_ctrl).any(axis=0)
        n_nan_cols = np.sum(nan_cols)
        # print(f'Num NaNs: {n_nan_cols=}')
        if n_nan_cols > 10:
            print(f'Lots! {n_nan_cols=}')
            return np.nan
        else:
            flat_ctrl = flat_ctrl[:, ~nan_cols]
            ERS_ctrl = ERS_ctrl[~nan_cols]
        flat_itr = np.array([1] * len(flat_ctrl))
        X = np.hstack([flat_itr[:, None], flat_ctrl])

        try:
            solution, residuals, rank, s = np.linalg.lstsq(X, flat_focus,
                                                           rcond=None)
        except np.linalg.LinAlgError as e:
            print(f'{e=}')
            print(f'{flat_focus=}')
            plt.imshow(flat_ctrl, aspect='auto')
            plt.show()
            quit()
        ERS_ctrl = np.transpose(ERS_ctrl, (1, 2, 0))
        ERS_focus -= np.dot(ERS_ctrl, solution[1:])

    ERS_sames = np.diag(ERS_focus)
    ERS_focus_ = ERS_focus.copy()
    ERS_focus_[np.eye(len(ERS_focus_), dtype=bool)] = np.nan
    ERS_elses = np.nanmean(ERS_focus_, axis=1)
    ERS_dif = ERS_sames - ERS_elses
    score = np.nanmean(ERS_dif)

    # same_M = np.nanmean(ERS_sames)
    else_M = np.nanmean(ERS_elses)
    var_explained = score / (1 - else_M)
    var_explained_sign = np.sign(var_explained)
    var_explained = (var_explained ** 2) * var_explained_sign

    return score, var_explained



def do_regr_RSA_sn(sn, ROI_focus_conn, ROIs_ctrl, fp, trial_similarity,
                   second_order, RDM_method, stdize_by_run, semantic,
                   fp0, fp1):

    # TODO: within to NaN

    # conn = ROI_focus.split('_')[-1]
    # ROI_focus = '_'.join(ROI_focus.split('_')[:-1])
    dir_in = fr'cache/conn_RSA/ars/RSA'
    dir_focus = (f'{dir_in}/{fp}_{trial_similarity}_'
                 f'{second_order}_{RDM_method}_{stdize_by_run}')
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus_conn}.npy'
    with open(fp_focus, 'rb') as f:
        RSM_focus = np.load(f)
        if RDM_method == 'within_nan':
            RSM_focus = within_run_to_nan(RSM_focus)

    flat_focus = RSM_focus[np.tril_indices_from(RSM_focus, k=-1)]
    fp_stim = f'{dir_focus}/{sn}_stim_{semantic}.npy'
    with open(fp_stim, 'rb') as f:
        RSM_stim = np.load(f)
    flat_stim = RSM_stim[np.tril_indices_from(RSM_stim, k=-1)]

    flat_itr = np.array([1] * len(flat_focus))

    if len(ROIs_ctrl):
        flat_ctrls = []
        skips = 0
        for ROI_ctrl in ROIs_ctrl:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                skips += 1
                continue
            with open(fp_ctrl, 'rb') as f:
                RSM_ctrl = np.load(f)
            flat_ctrl = RSM_ctrl[np.tril_indices_from(RSM_ctrl, k=-1)]
            flat_ctrls.append(flat_ctrl)
        flat_ctrl = np.array(flat_ctrls).T

        # flat_ctrl = ERS_ctrl.reshape(ERS_ctrl.shape[0], -1).T # (12996, 22)
        nan_cols = np.isnan(flat_ctrl).any(axis=0)
        n_nan_cols = np.sum(nan_cols) + skips
        if n_nan_cols > 10:
            print(f'Lots ({sn})! {n_nan_cols=}')
        else:
            flat_ctrl = flat_ctrl[:, ~nan_cols]

        X = np.hstack([flat_itr[:, None], flat_focus[:, None], flat_ctrl])
    else:
        X = np.hstack([flat_itr[:, None], flat_focus[:, None]])
    if RDM_method == 'within_nan':
        # print(len(X))
        X = X[~np.isnan(flat_focus), :]
        # print(len(X))
        # quit()
        flat_stim = flat_stim[~np.isnan(flat_focus)]
    # else:
    assert np.sum(np.isnan(X)) == 0
    # print(f'{X.shape=}')
    # quit()
    X[:, 1:] = stats.zscore(X[:, 1:], axis=0)
    flat_stim = stats.zscore(flat_stim)
    # print(flat_stim)
    # print(f'{X.shape=}')
    # print(X[:, 0])
    # print(X[:, 1])
    # quit()
    # plt.imshow(X, aspect='auto')
    # plt.show()
    solution, residuals, rank, s = np.linalg.lstsq(X, flat_stim, rcond=None)
    # print(f'{residuals}')
    # print(f'{solution=}')
    # print(f'{residuals=}')
    # plt.plot(flat_stim)
    # plt.show()
    r_sq = 1 - residuals / np.sum((flat_stim - np.mean(flat_stim)) ** 2)
    # print(f'{r_sq=}')
    # quit()
    return solution[1], r_sq


def do_regr_ISPC_sn(sn, ROI_focus_conn, ROIs_ctrl, fp, trial_similarity,
                   second_order, RDM_method, stdize_by_run, semantic,
                   fp0, fp1):


    dir_in = fr'cache/conn_RSA/ars/ISPC'
    dir_focus = f'{dir_in}/{fp}_{trial_similarity}'
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus_conn}.npy'
    with open(fp_focus, 'rb') as f:
        ISPC_focus = np.load(f)
    flat_focus = ISPC_focus.flatten()

    if len(ROIs_ctrl):
        ISPC_ctrls = []
        flat_ctrls = []
        skips = 0
        for ROI_ctrl in ROIs_ctrl:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                skips += 1
                continue
            with open(fp_ctrl, 'rb') as f:
                ISPC_ctrl = np.load(f)
            ISPC_ctrls.append(ISPC_ctrl)
            flat_ctrls.append(ISPC_ctrl.flatten())

        flat_ctrl = np.array(flat_ctrls).T
        assert flat_ctrl.shape[1] > 20, f'{flat_ctrl.shape=}'

        try:
            flat_itr = np.array([1] * len(flat_ctrl))
            X = np.hstack([flat_itr[:, None], flat_ctrl])
            solution, residuals, rank, s = np.linalg.lstsq(X, flat_focus,
                                                           rcond=None)
        except np.linalg.LinAlgError as e:
            print(f'{e=}')
            print(f'{flat_focus=}')
            plt.imshow(X[:, 1:], aspect='auto')
            plt.show()
            quit()
        ISPC_ctrls = np.array(ISPC_ctrls).T
        ISPC_focus -= np.dot(ISPC_ctrls, solution[1:])


    ERS_sames = np.diag(ISPC_focus)
    ERS_focus_ = ISPC_focus.copy()
    ERS_focus_[np.eye(len(ERS_focus_), dtype=bool)] = np.nan
    ERS_elses = np.nanmean(ERS_focus_, axis=1)
    ERS_dif = ERS_sames - ERS_elses
    score = np.nanmean(ERS_dif)

    # same_M = np.nanmean(ERS_sames)
    else_M = np.nanmean(ERS_elses)
    var_explained = score / (1 - else_M)
    var_explained_sign = np.sign(var_explained)
    var_explained = (var_explained ** 2) * var_explained_sign

    # score = np.mean(ROI_scores)
    return score, var_explained


def send_to_specific(kwargs, RSA, ISPC=False):
    fps = prep_fps('8')
    scores = []
    r_sqs = []
    kwargs['ROIs_ctrl'] = sorted(kwargs['ROIs_ctrl'])  # needed for pkl wrap

    if RSA:
        for fp in fps:
            kwargs['fp'] = fp
            score, r_sq = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                verbose=-1,
                                easy_override=False)
            scores.append(score)
            r_sqs.append(r_sq)
    elif ISPC:
        for fp in fps:
            kwargs['fp'] = fp
            score, r_sq = pickle_wrap(do_regr_ISPC_sn, kwargs=kwargs,
                                verbose=-1,
                                easy_override=False)
            scores.append(score)
            r_sqs.append(r_sq)
    else:
        for fp0 in fps:
            for fp1 in fps:
                if fp0 >= fp1: continue
                kwargs['fp0'] = fp0
                kwargs['fp1'] = fp1
                # score = do_regr_ERS_sn(**kwargs)
                score, r_sq = pickle_wrap(do_regr_ERS_sn, kwargs=kwargs,
                                    verbose=-1)
                scores.append(score)
                r_sqs.append(r_sq)

    return np.nanmean(scores), np.nanmean(r_sqs)

def run_all_sn(kwargs, RSA, ISPC):
    t_st = time()
    z_l = []
    r_sqs = []
    sns = get_sns('all')['healthy']
    for sn in tqdm(sns, desc='conn regressing...', position=0, leave=True):
        if sn in ['138', '224']: continue
        kwargs['sn'] = sn

        try:
            z, r_sq = send_to_specific(kwargs, RSA, ISPC)
        except FileNotFoundError as e:
            # print(f'sn: {e}')
            continue
        if np.isnan(z):
            continue
        z_l.append(z)
        r_sqs.append(r_sq)
    # print(f'{time() - t_st:.2f} s')
    t, p = stats.ttest_1samp(z_l, 0)
    print(f't[{len(z_l)-1}]={t:.3f}, {p=:.3f}')
    return np.sqrt(np.nanmean(np.array(r_sqs)))

def plot_r_sqs(kwargs, RSA, ISPC):
    r_sqs_conn_ROIs = run_all_sn(kwargs, RSA, ISPC)

    ROI_bold = '_'.join(kwargs['ROI_focus_conn'].split('_')[:-1]) + '_BOLD'
    kwargs_ = kwargs.copy()
    kwargs_['ROIs_ctrl'] = kwargs_['ROIs_ctrl'] + [ROI_bold]
    r_sqs_conn_all = run_all_sn(kwargs, RSA, ISPC)

    kwargs_ = kwargs.copy()
    kwargs_['ROIs_ctrl'] = [ROI_bold]
    r_sqs_conn_bold = run_all_sn(kwargs_, RSA, ISPC)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus_conn'] = ROI_bold
    r_sqs_bold_ROI = run_all_sn(kwargs_, RSA, ISPC)

    kwargs_ = kwargs.copy()
    print(f'{kwargs_=}')
    kwargs_['ROIs_ctrl'] = []
    r_sqs_conn = run_all_sn(kwargs_, RSA, ISPC)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus_conn'] = ROI_bold
    print(f'{kwargs_=}')
    kwargs_['ROIs_ctrl'] = []
    r_sqs_bold = run_all_sn(kwargs_, RSA, ISPC)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus_conn'] = kwargs_['ROIs_ctrl'][0]
    kwargs_['ROIs_ctrl'] = kwargs_['ROIs_ctrl'][1:]
    r_sqs_ROIs = run_all_sn(kwargs_, RSA, ISPC)

    print(f'{r_sqs_conn_all=:.4f}')
    print(f'\t{r_sqs_conn_ROIs=:.4f}')
    print(f'\t{r_sqs_conn_bold=:.4f}')
    print(f'\t{r_sqs_bold_ROI=:.4f}')
    print(f'\t{r_sqs_conn=:.4f}')
    print(f'\t{r_sqs_bold=:.4f}')
    print(f'\t{r_sqs_ROIs=:.4f}')

def do_regr():
    ISPC = False
    RSA = True
    semantic = False
    conn = 'prod'
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    stdize_by_run = True if trial_similarity == 'euc' else False
    target_ROI = 'PFC_ACC'

    ROI_focus = f'{target_ROI}_{conn}'

    regions = set(prep_networks(
        network_setting=ROI2NETWORK[target_ROI])[target_ROI])
    ROIs_match = [ROI for region in regions
                      for ROI in get_BNA_ROIs() if region in ROI]
    ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]
    ROIs_ctrl = ROI_lvl_control
    # ROI_focus = ROI_lvl_control[0]
    # ROIs_ctrl = []

    kwargs = {'semantic': semantic,
              'ROI_focus_conn': ROI_focus, 'ROIs_ctrl': ROIs_ctrl,
              'fp': None, #'bl8_fMRI',
              'fp0': None, #'bl8_fMRI',
              'fp1': None, #'obj8_fMRI',
              'trial_similarity': trial_similarity,
              'second_order': second_order, 'RDM_method': RDM_method,
              'stdize_by_run': stdize_by_run,}
    print(f'Num controls: {len(ROIs_ctrl)}')
    # run_all_sn(kwargs, RSA, ISPC)
    plot_r_sqs(kwargs, RSA, ISPC)


if __name__ == '__main__':
    do_regr()











