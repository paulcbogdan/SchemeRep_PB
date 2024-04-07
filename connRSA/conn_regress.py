import os

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from connRSA.conn_analyze_IRAFs import prep_network2ROI, ROI2NETWORK
from connRSA.conn_utils import get_BNA_ROIs
from connRSA.single_trial_conn import prep_fps
from fMRI_proc import within_run_to_nan, get_IRAFs
from old.networks import prep_networks
from organize_bhv import get_trial_info
from utils import pickle_wrap
import statsmodels.formula.api as smf

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

import numpy as np
import scipy.stats as stats

from time import time
from org_sns import get_sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold

def do_RSM_ERS_sn(sn, ROI_focus_conn, ROIs_ctrl,
                  fp0, fp1, trial_similarity, stdize_by_run,
                  semantic, second_order, RDM_method, fp,
                  regress_row=False):

    dir_in = fr'cache/conn_RSA/ars/RSA'
    RDM_method_ = 'within_nan' if RDM_method == 'double_nan' else RDM_method
    dir_focus1 = (f'{dir_in}/{fp0}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus1 = f'{dir_focus1}/{sn}_{ROI_focus_conn}.npy'
    with open(fp_focus1, 'rb') as f:
        RSM_focus1 = np.load(f)
        if 'nan' in RDM_method:
            RSM_focus1 = within_run_to_nan(RSM_focus1)

    flat_focus = RSM_focus1[np.tril_indices_from(RSM_focus1, k=-1)]


    dir_focus2 = (f'{dir_in}/{fp1}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus2 = f'{dir_focus2}/{sn}_{ROI_focus_conn}.npy'
    with open(fp_focus2, 'rb') as f:
        RSM_stim = np.load(f)
        if 'double_nan' in RDM_method:
            RSM_stim = within_run_to_nan(RSM_stim)
            # plt.imshow(RSM_stim)
            # plt.show()


        df_sn = get_trial_info(sn, verbose=-1)
        sess0 = fp0.split('_')[0][:-1]
        sess1 = fp1.split('_')[0][:-1]
        map1_to_0 = {}
        assert RSM_stim.shape == (114, 114)

        df_sn[f'{sess0}_trial'] -= 1
        df_sn[f'{sess1}_trial'] -= 1
        if df_sn[f'{sess1}_trial'].max() > 113:
            df_sn_ = df_sn.sort_values(f'{sess1}_trial')
            mapper = {}
            for i, (_, row) in enumerate(df_sn_.iterrows()):
                mapper[row[f'{sess1}_trial']] = i
            df_sn[f'{sess1}_trial'] = df_sn[f'{sess1}_trial'].map(mapper)
        if df_sn[f'{sess0}_trial'].max() > 113:
            df_sn_ = df_sn.sort_values(f'{sess0}_trial')
            mapper = {}
            for i, (_, row) in enumerate(df_sn_.iterrows()):
                mapper[row[f'{sess0}_trial']] = i
            df_sn[f'{sess0}_trial'] = df_sn[f'{sess0}_trial'].map(mapper)


        # print(df_sn[f'{sess1}_trial'].sort_values())

        # print(df_sn[f'{sess1}_trial'].min())
        # print(df_sn[f'{sess1}_trial'].max())
        # quit()
        for idx, row in df_sn.iterrows():
            map1_to_0[row[f'{sess1}_trial']] = row[f'{sess0}_trial']
        RSM_stim_ = np.zeros((114, 114))
        for i in range(114):
            for j in range(114):
                new_i = map1_to_0[i]
                new_j = map1_to_0[j]
                RSM_stim_[new_i, new_j] = RSM_stim[i, j]
        RSM_stim = RSM_stim_
        # plt.imshow(RSM_stim)
        # plt.show()
        # quit()



    flat_stim = RSM_stim[np.tril_indices_from(RSM_stim, k=-1)]
    flat_itr = np.array([1] * len(flat_focus))
    if len(ROIs_ctrl):
        flat_ctrls = []
        skips = 0
        for ROI_ctrl in ROIs_ctrl:
            fp_ctrl = f'{dir_focus1}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                skips += 1
                continue
            with open(fp_ctrl, 'rb') as f:
                RSM_ctrl = np.load(f)
            flat_ctrl = RSM_ctrl[np.tril_indices_from(RSM_ctrl, k=-1)]
            flat_ctrls.append(flat_ctrl)
        flat_ctrl = np.array(flat_ctrls).T

        nan_cols = np.isnan(flat_ctrl).any(axis=0)
        n_nan_cols = np.sum(nan_cols) + skips
        if n_nan_cols > 10:
            print(f'Lots ({sn})! {n_nan_cols=}')
        else:
            flat_ctrl = flat_ctrl[:, ~nan_cols]

        X = np.hstack([flat_itr[:, None], flat_focus[:, None], flat_ctrl])
    else:
        X = np.hstack([flat_itr[:, None], flat_focus[:, None]])
    if 'within_nan' == RDM_method:
        X = X[~np.isnan(flat_focus), :]
        flat_stim = flat_stim[~np.isnan(flat_focus)]
    elif 'double_nan' == RDM_method:
        goods = ~(np.isnan(flat_focus) | np.isnan(flat_stim))
        X = X[goods, :]
        flat_stim = flat_stim[goods]


    assert np.sum(np.isnan(X)) == 0

    solution, residuals, rank, s = np.linalg.lstsq(X, flat_stim, rcond=None)
    p = X.shape[1] - 1
    n = X.shape[0] - 1 # minus 1 because of the intercept
    r_sq = 1 - residuals / np.sum((flat_stim - np.mean(flat_stim)) ** 2)
    r_sq = r_sq[0]
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p - 1)
    return solution[1], r_sq






def do_regr_ERS_sn(sn, ROI_focus_conn, ROIs_ctrl,
                   fp0, fp1, trial_similarity, stdize_by_run,
                   semantic, second_order, RDM_method, fp,
                   regress_row=True):
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
                ERSs_ctrl = np.load(f)
            ERS_ctrl_l.append(ERSs_ctrl)
        ERSs_ctrl = np.array(ERS_ctrl_l)
        flat_ctrl = ERSs_ctrl.reshape(ERSs_ctrl.shape[0], -1).T # (12996, 22)
        nan_cols = np.isnan(flat_ctrl).any(axis=0)
        n_nan_cols = np.sum(nan_cols)
        if n_nan_cols > 10:
            print(f'Lots! {n_nan_cols=}')
            return np.nan
        else:
            flat_ctrl = flat_ctrl[:, ~nan_cols]
            ERSs_ctrl = ERSs_ctrl[~nan_cols]

        if regress_row:
            ERS_dif_focus, var_explained_focus = get_ERS_scores(ERS_focus)
            ERS_difs_ctrls, var_explaineds_ctrls = [], []
            for ERS_ctrl in ERS_ctrl_l:
                ERS_dif_ctrl, var_explained_ctrl = get_ERS_scores(ERS_ctrl)
                ERS_difs_ctrls.append(ERS_dif_ctrl)
                var_explaineds_ctrls.append(var_explained_ctrl)
            ERS_difs_ctrls = np.array(ERS_difs_ctrls).T
            X = np.hstack([np.ones((114, 1)), ERS_difs_ctrls])
            solution, residuals, rank, s = np.linalg.lstsq(X, ERS_dif_focus,
                                                           rcond=None)
            ERS_dif_focus -= np.dot(ERS_difs_ctrls, solution[1:])

            var_explaineds_ctrls = np.array(var_explaineds_ctrls).T
            X = np.hstack([np.ones((114, 1)), var_explaineds_ctrls])
            solution, residuals, rank, s \
                = np.linalg.lstsq(X, var_explained_focus, rcond=None)
            var_explained_focus -= np.dot(var_explaineds_ctrls, solution[1:])
            return np.nanmean(ERS_dif_focus), np.nanmean(var_explained_focus)

        else:
            flat_itr = np.array([1] * len(flat_ctrl))
            X = np.hstack([flat_itr[:, None], flat_ctrl])
            solution, residuals, rank, s = np.linalg.lstsq(X, flat_focus,
                                                           rcond=None)
            ERSs_ctrl = np.transpose(ERSs_ctrl, (1, 2, 0))
            ERS_focus -= np.dot(ERSs_ctrl, solution[1:])

    ERS_dif, var_explained = get_ERS_scores(ERS_focus)
    return ERS_dif, var_explained

    # ERS_sames = np.diag(ERS_focus)
    # ERS_focus_ = ERS_focus.copy()
    # ERS_focus_[np.eye(len(ERS_focus_), dtype=bool)] = np.nan
    # ERS_elses = np.nanmean(ERS_focus_, axis=1)
    # ERS_dif = ERS_sames - ERS_elses
    #
    # score = np.nanmean(ERS_dif)
    #
    # # same_M = np.nanmean(ERS_sames)
    # else_M = np.nanmean(ERS_elses)
    # var_explained = score / (1 - else_M)
    # var_explained_sign = np.sign(var_explained)
    # var_explained = (var_explained ** 2) * var_explained_sign
    #
    # return score, var_explained

def get_ERS_scores(ERS_mat):
    ERS_sames = np.diag(ERS_mat)
    ERS_focus_ = ERS_mat.copy()
    ERS_focus_[np.eye(len(ERS_focus_), dtype=bool)] = np.nan
    ERS_elses = np.nanmean(ERS_focus_, axis=1)
    ERS_dif = ERS_sames - ERS_elses

    var_explained = ERS_dif / (1 - ERS_elses)
    var_explained_sign = np.sign(var_explained)
    var_explained = (var_explained ** 2) * var_explained_sign


    return ERS_dif, var_explained


def do_regr_RSA_sn(sn, ROI_focus_conn, ROIs_ctrl, fp, trial_similarity,
                   second_order, RDM_method, stdize_by_run, semantic,
                   fp0, fp1, cv=False,
                   regress_row=True):

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
        flat_ctrl_l = []
        RSM_ctrl_l = []
        skips = 0
        for ROI_ctrl in ROIs_ctrl:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                skips += 1
                continue
            with open(fp_ctrl, 'rb') as f:
                RSM_ctrl = np.load(f)
                RSM_ctrl_l.append(RSM_ctrl)
            flat_ctrls = RSM_ctrl[np.tril_indices_from(RSM_ctrl, k=-1)]
            flat_ctrl_l.append(flat_ctrls)
        flat_ctrls = np.array(flat_ctrl_l).T

        nan_cols = np.isnan(flat_ctrls).any(axis=0)
        n_nan_cols = np.sum(nan_cols) + skips
        if n_nan_cols > 10:
            print(f'Lots ({sn})! {n_nan_cols=}')
        else:
            flat_ctrls = flat_ctrls[:, ~nan_cols]


        if regress_row:
            IRAFs_ctrl = []
            for RSM_ctrl in RSM_ctrl_l:
                IRAFs = get_IRAFs(RSM_stim, RSM_ctrl, None,
                                  within_to_nan=RDM_method == 'within_nan',
                                  by_run=False, second_order='corr')
                IRAFs_ctrl.append(IRAFs)

            IRAFs_focus = get_IRAFs(RSM_stim, RSM_focus, None,
                                    within_to_nan=RDM_method == 'within_nan',
                                    by_run=False, second_order='corr')
            X = np.hstack([np.ones((114, 1)), np.array(IRAFs_ctrl).T])
            solution, residuals, rank, s = np.linalg.lstsq(X, IRAFs_focus,
                                                           rcond=None)
            IRAFs_focus -= np.dot(np.array(IRAFs_ctrl).T, solution[1:])

            M = np.nanmean(IRAFs_focus)
            return M, (M ** 2) * np.sign(M)

        X = np.hstack([flat_itr[:, None], flat_focus[:, None], flat_ctrls])
    else:
        X = np.hstack([flat_itr[:, None], flat_focus[:, None]])
    if RDM_method == 'within_nan':
        X = X[~np.isnan(flat_focus), :]
        flat_stim = flat_stim[~np.isnan(flat_focus)]
    assert np.sum(np.isnan(X)) == 0

    solution, residuals, rank, s = np.linalg.lstsq(X, flat_stim, rcond=None)
    p = X.shape[1] - 1
    n = X.shape[0] - 1 # minus 1 because of the intercept
    r_sq = 1 - residuals / np.sum((flat_stim - np.mean(flat_stim)) ** 2)
    r_sq = r_sq[0]
    # print(f'{r_sq=:.4f}')
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p - 1)
    # print(f'Number of predictors: {X.shape[1]=}')
    return solution[1], r_sq


def do_regr_ISPC_sn(sn, ROI_focus_conn, ROIs_ctrl, fp, trial_similarity,
                   second_order, RDM_method, stdize_by_run, semantic,
                   fp0, fp1, regress_row=False):

    dir_in = fr'cache/conn_RSA/ars/ISPC'
    dir_focus = f'{dir_in}/{fp}_{trial_similarity}'
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus_conn}.npy'
    with open(fp_focus, 'rb') as f:
        ISPC_focus = np.load(f)
    flat_focus = ISPC_focus.flatten()

    if len(ROIs_ctrl):
        ISPC_ctrls_l = []
        flat_ctrls = []
        skips = 0
        for ROI_ctrl in ROIs_ctrl:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                skips += 1
                # print(f'{sn}: missing')
                continue
            with open(fp_ctrl, 'rb') as f:
                ISPC_ctrl = np.load(f)
            ISPC_ctrls_l.append(ISPC_ctrl)
            flat_ctrls.append(ISPC_ctrl.flatten())

        ERSs_ctrl = np.array(ISPC_ctrls_l)
        flat_ctrl = np.array(flat_ctrls).T
        if len(flat_ctrl) == 0:
            return np.nan, np.nan

        if regress_row:
            ERS_dif_focus, var_explained_focus = get_ERS_scores(ISPC_focus)
            ERS_difs_ctrls, var_explaineds_ctrls = [], []
            for ERS_ctrl in ISPC_ctrls_l:
                ERS_dif_ctrl, var_explained_ctrl = get_ERS_scores(ERS_ctrl)
                ERS_difs_ctrls.append(ERS_dif_ctrl)
                var_explaineds_ctrls.append(var_explained_ctrl)
            ERS_difs_ctrls = np.array(ERS_difs_ctrls).T
            X = np.hstack([np.ones((114, 1)), ERS_difs_ctrls])
            solution, residuals, rank, s = np.linalg.lstsq(X, ERS_dif_focus,
                                                           rcond=None)
            ERS_dif_focus -= np.dot(ERS_difs_ctrls, solution[1:])

            var_explaineds_ctrls = np.array(var_explaineds_ctrls).T
            X = np.hstack([np.ones((114, 1)), var_explaineds_ctrls])
            solution, residuals, rank, s \
                = np.linalg.lstsq(X, var_explained_focus, rcond=None)
            var_explained_focus -= np.dot(var_explaineds_ctrls, solution[1:])
            return np.nanmean(ERS_dif_focus), np.nanmean(var_explained_focus)
        else:
            flat_itr = np.array([1] * len(flat_ctrl))
            X = np.hstack([flat_itr[:, None], flat_ctrl])
            solution, residuals, rank, s = np.linalg.lstsq(X, flat_focus,
                                                           rcond=None)
            ERSs_ctrl = np.transpose(ERSs_ctrl, (1, 2, 0))
            ISPC_focus -= np.dot(ERSs_ctrl, solution[1:])

    ERS_dif, var_explained = get_ERS_scores(ISPC_focus)
    return ERS_dif, var_explained

    # ERS_sames = np.diag(ISPC_focus)
    # ERS_focus_ = ISPC_focus.copy()
    # ERS_focus_[np.eye(len(ERS_focus_), dtype=bool)] = np.nan
    # ERS_elses = np.nanmean(ERS_focus_, axis=1)
    # ERS_dif = ERS_sames - ERS_elses
    # score = np.nanmean(ERS_dif)
    #
    # # same_M = np.nanmean(ERS_sames)
    # else_M = np.nanmean(ERS_elses)
    # var_explained = score / (1 - else_M)
    # var_explained_sign = np.sign(var_explained)
    # var_explained = (var_explained ** 2) * var_explained_sign
    #
    # # score = np.mean(ROI_scores)
    # return score, var_explained


def send_to_specific(kwargs, RSA, ISPC=False, ERS_alt=False, regress_row=True):
    fps = prep_fps('8')
    scores = []
    r_sqs = []
    kwargs['ROIs_ctrl'] = sorted(kwargs['ROIs_ctrl'])  # needed for pkl wrap

    if RSA:
        for fp in fps:
            kwargs['fp'] = fp
            kwargs['cv'] = False
            kwargs['regress_row'] = regress_row
            score, r_sq = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                      verbose=-1, easy_override=False)
            scores.append(score)
            r_sqs.append(r_sq)
    elif ISPC:
        for fp in fps:
            kwargs['fp'] = fp
            kwargs['regress_row'] = regress_row
            score, r_sq = pickle_wrap(do_regr_ISPC_sn, kwargs=kwargs,
                                      verbose=-1, easy_override=False)
            scores.append(score)
            r_sqs.append(r_sq)
    else:
        for fp0 in fps:
            for fp1 in fps:
                if fp0 >= fp1: continue
                kwargs['fp0'] = fp0
                kwargs['fp1'] = fp1
                kwargs['regress_row'] = regress_row
                f = do_RSM_ERS_sn if ERS_alt else do_regr_ERS_sn

                score, r_sq = pickle_wrap(f, kwargs=kwargs,
                                          verbose=-1,
                                          easy_override=True)
                scores.append(score)
                r_sqs.append(r_sq)

    return np.nanmean(scores), np.nanmean(r_sqs)

def run_all_sn(kwargs, RSA, ISPC, ERS_alt):
    # t_st = time()
    z_l = []
    r_sqs = []
    sns = get_sns('all')['healthy']
    num_predictors = 1 + len(kwargs['ROIs_ctrl'])
    preds = [kwargs['ROI_focus_conn']] + kwargs['ROIs_ctrl']
    print('-')
    print(f'{preds=}')
    print(f'{num_predictors=}')
    for sn in sns:#, desc='conn regressing...', position=0, leave=False):
        if sn in ['138', '224']: continue
        kwargs['sn'] = sn

        try:
            z, r_sq = send_to_specific(kwargs, RSA, ISPC, ERS_alt)
        except FileNotFoundError as e:
            # print(f'sn: {e}')
            continue
        if np.isnan(z):
            continue
        z_l.append(z)
        r_sqs.append(r_sq)
    # r_sqs = np.array(r_sqs) * 100
    # print(f'{time() - t_st:.2f} s')
    # print(f'{r_sqs=}')
    # print(np.nanmean(np.array(r_sqs)))
    # plt.hist(r_sqs)
    # plt.show()
    t, p = stats.ttest_1samp(z_l, 0)
    M_r_sq = np.nanmean(np.array(r_sqs))
    print(f't[{len(z_l)-1}]={t:.3f}, {p=:.3f} | {M_r_sq=:.5%}')
    return t, np.nanmean(np.array(r_sqs))

def plot_stacked_bars(kwargs, RSA, ISPC, ERS_alt):
    ROI_bold = '_'.join(kwargs['ROI_focus_conn'].split('_')[:-1]) + '_BOLD'

    kwargs_ = kwargs.copy()
    kwargs_['ROIs_ctrl'] = kwargs_['ROIs_ctrl'] + [ROI_bold]
    t_conn_all, r_sqs_conn_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt)
    kwargs_['ROIs_ctrl'] = []
    t_conn, r_sqs_conn = run_all_sn(kwargs_, RSA, ISPC, ERS_alt)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus_conn'] = kwargs_['ROIs_ctrl'][0]
    kwargs_['ROIs_ctrl'] = [kwargs['ROI_focus_conn'], ROI_bold]
    t_ROIs_all, r_sqs_ROIs_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt)
    kwargs_['ROIs_ctrl'] = []
    t_ROIs, r_sqs_ROIs = run_all_sn(kwargs_, RSA, ISPC, ERS_alt)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus_conn'] = ROI_bold
    kwargs_['ROIs_ctrl'] = kwargs_['ROIs_ctrl'] + [kwargs['ROI_focus_conn']]
    t_bold_all, r_sqs_bold_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt)
    kwargs_['ROIs_ctrl'] = []
    t_bold, r_sqs_bold = run_all_sn(kwargs_, RSA, ISPC, ERS_alt)

    print('-')
    print(f'{t_conn=:.3f} ({r_sqs_conn:.4f})')
    print(f'\t{t_conn_all=:.3f} ({r_sqs_conn_all:.4f})')
    print(f'{t_ROIs=:.3f} ({r_sqs_ROIs:.4f})')
    print(f'\t{t_ROIs_all=:.3f} ({r_sqs_ROIs_all:.4f})')
    print(f'{t_bold=:.3f} ({r_sqs_bold:.4f})')
    print(f'\t{t_bold_all=:.3f} ({r_sqs_bold_all:.4f})')

    # print(f'\t{r_sqs_conn_all=:.4f}')
    # print(f'{r_sqs_ROIs=:.4f}')
    # print(f'\t{r_sqs_ROIs_all=:.4f}')
    # print(f'{r_sqs_bold=:.4f}')
    # print(f'\t{r_sqs_bold_all=:.4f}')



def plot_r_sqs(kwargs, RSA, ISPC):

    ROI_bold = '_'.join(kwargs['ROI_focus_conn'].split('_')[:-1]) + '_BOLD'
    kwargs_ = kwargs.copy()
    kwargs_['ROIs_ctrl'] = kwargs_['ROIs_ctrl'] + [ROI_bold]
    r_sqs_conn_all = run_all_sn(kwargs_, RSA, ISPC)

    r_sqs_conn_ROIs = run_all_sn(kwargs, RSA, ISPC)

    kwargs_ = kwargs.copy()
    kwargs_['ROIs_ctrl'] = [ROI_bold]
    r_sqs_conn_bold = run_all_sn(kwargs_, RSA, ISPC)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus_conn'] = ROI_bold
    r_sqs_bold_ROI = run_all_sn(kwargs_, RSA, ISPC)

    kwargs_ = kwargs.copy()
    kwargs_['ROIs_ctrl'] = []
    r_sqs_conn = run_all_sn(kwargs_, RSA, ISPC)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus_conn'] = ROI_bold
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

def prep_ROI_avg(target_name, ROI_cols,
                 fp0, fp1, trial_similarity, stdize_by_run, semantic,
                 second_order, RDM_method, fp, RSA=False, ISPC=False,
                 ERS_alt=False):
    kwargs = locals().copy()
    if RSA or ISPC or ERS_alt:
        if fp is None:
            fps = prep_fps('8')
            for fp in fps:
                kwargs['fp'] = fp
                prep_ROI_avg(**kwargs)
            return
    else:
        if fp0 is None:
            fps = prep_fps('8')
            for fp0 in fps:
                for fp1 in fps:
                    if fp0 >= fp1: continue
                    kwargs['fp0'] = fp0
                    kwargs['fp1'] = fp1
                    prep_ROI_avg(**kwargs)
            return

    if RSA:
        dir_in = fr'cache/conn_RSA/ars/RSA'
        dir_focus = (f'{dir_in}/{fp}_{trial_similarity}_'
                     f'{second_order}_{RDM_method}_{stdize_by_run}')
    elif ISPC:
        dir_in = fr'cache/conn_RSA/ars/ISPC'
        dir_focus = f'{dir_in}/{fp}_{trial_similarity}'
    else:
        dir_focus = fr'cache/conn_RSA/ars/ERS'
        dir_focus = (fr'{dir_focus}/{fp0}_{fp1}_{trial_similarity}_'
                     fr'{stdize_by_run}')

    sns = get_sns('all')['healthy']
    for sn in tqdm(sns, desc=f'Averaging ROI data: {target_name}',
                   position=0, leave=False):
        fp_out = f'{dir_focus}/{sn}_{target_name}.npy'
        if os.path.isfile(fp_out):
            continue

        skips = 0
        l = []
        for ROI_ctrl in ROI_cols:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                skips += 1
                continue
            with open(fp_ctrl, 'rb') as f:
                ISPC_ctrl = np.load(f)
                l.append(ISPC_ctrl)
        # if skips > 1:
        #     print(f'{skips=}')
        if len(l) == 0:
            continue

        ctrl_M = np.nanmean(np.array(l), axis=0)
        with open(fp_out, 'wb') as f:
            np.save(f, ctrl_M)


def do_regr():
    ISPC = False
    RSA = False
    semantic = False
    ERS_alt = False
    conn = 'prod'
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    stdize_by_run = True if trial_similarity == 'euc' else False
    # stdize_by_run = True
    # TODO: update ISPC to have stdize_by_run as a toggle
    # target_ROI = 'Occipital'
    target_ROI = 'MTL'
    # target_ROI = 'PFC_ACC'

    ROI_focus = f'{target_ROI}_{conn}'

    regions = set(prep_networks(
        network_setting=ROI2NETWORK[target_ROI])[target_ROI])
    ROIs_match = [ROI for region in regions
                      for ROI in get_BNA_ROIs() if region in ROI]
    ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]

    kwargs = {'semantic': semantic, 'fp': None, 'fp0': None, 'fp1': None,
              'trial_similarity': trial_similarity,
              'second_order': second_order, 'RDM_method': RDM_method,
              'stdize_by_run': stdize_by_run,}

    target_name = fr'{target_ROI}_M'
    prep_ROI_avg(target_name, ROI_lvl_control, RSA=RSA, ISPC=ISPC,
                 ERS_alt=ERS_alt,
                 **kwargs)
    kwargs['ROI_focus_conn'] = ROI_focus
    kwargs['ROIs_ctrl'] = [target_name]

    plot_stacked_bars(kwargs, RSA, ISPC, ERS_alt)

import sys
sys.setrecursionlimit(10000)

if __name__ == '__main__':
    do_regr()


