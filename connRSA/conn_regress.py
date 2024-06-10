import os
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.conn_analyze_IRAFs import prep_network2ROI, ROI2NETWORK
from connRSA.conn_utils import get_BNA_ROIs
from connRSA.single_trial_conn import prep_fps
from fMRI_proc import within_run_to_nan, get_IRAFs
from old.networks import prep_networks
from old.plot_gen import my_plot_surf
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


def do_RSM_ERS_sn(sn, ROI_focus, ROIs_ctrl,
                  fp0, fp1, trial_similarity, stdize_by_run,
                  semantic, second_order, RDM_method, fp,
                  four_tasks,
                  regress_row=False):

    dir_in = fr'cache/conn_RSA/ars/RSA'
    RDM_method_ = 'within_nan' if RDM_method == 'double_nan' else RDM_method
    dir_focus1 = (f'{dir_in}/{fp0}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus1 = f'{dir_focus1}/{sn}_{ROI_focus}.npy'

    with open(fp_focus1, 'rb') as f:
        RSM_focus1 = np.load(f)
        if 'nan' in RDM_method:
            RSM_focus1 = within_run_to_nan(RSM_focus1)

    flat_focus = RSM_focus1[np.tril_indices_from(RSM_focus1, k=-1)]


    cmb = '_cmb' if 'cmb' in ROI_focus else ''
    dir_focus2 = (f'{dir_in}/{fp1}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus2 = f'{dir_focus2}/{sn}_{ROI_focus}{cmb}.npy'
    with open(fp_focus2, 'rb') as f:
        RSM_focus2 = np.load(f)
        if 'double_nan' in RDM_method:
            RSM_focus2 = within_run_to_nan(RSM_focus2)

        df_sn = get_trial_info(sn, verbose=-1)
        sess0 = fp0.split('_')[0][:-1]
        sess1 = fp1.split('_')[0][:-1]
        map1_to_0 = {}
        assert RSM_focus2.shape == (114, 114)

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

        for idx, row in df_sn.iterrows():
            map1_to_0[row[f'{sess1}_trial']] = row[f'{sess0}_trial']
        RSM_focus2_ = np.zeros((114, 114))
        for i in range(114):
            for j in range(114):
                new_i = map1_to_0[i]
                new_j = map1_to_0[j]
                RSM_focus2_[new_i, new_j] = RSM_focus2[i, j]
        RSM_focus2 = RSM_focus2_


    flat_stim = RSM_focus2[np.tril_indices_from(RSM_focus2, k=-1)]
    flat_itr = np.array([1] * len(flat_focus))
    if len(ROIs_ctrl):
        flat_ctrls = []
        skips = 0
        for ROI_ctrl in ROIs_ctrl:
            cmb = '_cmb' if 'cmb' in ROIs_ctrl else ''
            fp_ctrl = f'{dir_focus1}/{sn}_{ROI_ctrl}{cmb}.npy'
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

def do_regr_ERS_sn(sn, ROI_focus, ROIs_ctrl, fp0, fp1, trial_similarity,
                   stdize_by_run, semantic, second_order, RDM_method, fp,
                   four_tasks, regress_row=True):
    dir_focus = fr'cache/conn_RSA/ars/ERS'
    dir_focus = fr'{dir_focus}/{fp0}_{fp1}_{trial_similarity}_{stdize_by_run}'
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus}.npy'
    try:
        with open(fp_focus, 'rb') as f:
            # age = os.path.getmtime(fp_focus)
            # from datetime import datetime
            # age = datetime.fromtimestamp(age)
            # print(f'{age} | {fp_focus=}')
            ERS_focus = np.load(f)
    except ValueError:
        print(f'Bad {sn}: {fp_focus=}')
        return np.nan, np.nan
    flat_focus = ERS_focus.flatten()

    if len(ROIs_ctrl):
        ERS_ctrl_l = []
        for ROI_ctrl in ROIs_ctrl:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            try:
                with open(fp_ctrl, 'rb') as f:
                    ERSs_ctrl = np.load(f)
            except ValueError as e:
                print(f'Bad allow_pickle problem {sn}: {fp_ctrl=}')
                quit()
                return np.nan, np.nan
            ERS_ctrl_l.append(ERSs_ctrl)
        ERSs_ctrl = np.array(ERS_ctrl_l)
        flat_ctrl = ERSs_ctrl.reshape(ERSs_ctrl.shape[0], -1).T # (12996, 22)
        nan_cols = np.isnan(flat_ctrl).any(axis=0)

        n_nan_cols = np.sum(nan_cols)
        if n_nan_cols > 10:
            print(f'Lots! {n_nan_cols=}')
            return np.nan, np.nan
        else:
            flat_ctrl = flat_ctrl[:, ~nan_cols]
            ERSs_ctrl = ERSs_ctrl[~nan_cols]
        # print(f'{nan_cols=}')
        if regress_row:
            ERS_dif_focus, var_explained_focus = get_ERS_scores(ERS_focus)
            ERS_difs_ctrls, var_explaineds_ctrls = [], []
            for ERS_ctrl in ERS_ctrl_l:
                ERS_dif_ctrl, var_explained_ctrl = get_ERS_scores(ERS_ctrl)
                ERS_difs_ctrls.append(ERS_dif_ctrl)
                var_explaineds_ctrls.append(var_explained_ctrl)
            ERS_difs_ctrls = np.array(ERS_difs_ctrls).T
            X = np.hstack([np.ones((114, 1)), ERS_difs_ctrls])
            if X.shape[1] == 1:
                print(f'Bad ({sn}): {X.shape=}')
                return np.nan, np.nan
            print(f'{X.shape=}')
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
            # print(f'{sn=}')
            # print(f'{X=}')
            if X.shape[1] == 1:
                print(f'Bad ({sn}): {X.shape=}')
                return np.nan, np.nan
            assert X.shape[1] > 1, f'Bad: {X.shape=}'
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

def get_ERS_scores(ERS_mat, get_same=False):
    ERS_sames = np.diag(ERS_mat)

    if get_same:
        return ERS_sames, None
    ERS_focus_ = ERS_mat.copy()
    ERS_focus_[np.eye(len(ERS_focus_), dtype=bool)] = np.nan
    ERS_elses = np.nanmean(ERS_focus_, axis=1)
    ERS_dif = ERS_sames - ERS_elses
    # M = np.nanmean(ERS_dif)
    # print(f'{M=:.3f}')
    # test = stats.spearmanr(ERS_sames, ERS_elses, nan_policy='omit')
    # print(f'{test=}')

    var_explained = ERS_dif / (1 - ERS_elses)
    var_explained_sign = np.sign(var_explained)
    var_explained = (var_explained ** 2) * var_explained_sign

    return ERS_dif, var_explained

def do_regr_RSA_sn(sn, ROI_focus, ROIs_ctrl, fp, trial_similarity,
                   second_order, RDM_method, stdize_by_run, semantic,
                   fp0, fp1, four_tasks, cv=False,
                   regress_row=True):

    dir_in = fr'cache/conn_RSA/ars/RSA'
    dir_focus = (f'{dir_in}/{fp}_{trial_similarity}_'
                 f'{second_order}_{RDM_method}_{stdize_by_run}')
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus}.npy'
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
        n_skip_cols = 0
        for ROI_ctrl in ROIs_ctrl:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                n_skip_cols += 1
                print(f'Missing ctrl ({sn}): {fp_ctrl=}')
                continue
            with open(fp_ctrl, 'rb') as f:
                RSM_ctrl = np.load(f)
                RSM_ctrl_l.append(RSM_ctrl)
            flat_ctrls = RSM_ctrl[np.tril_indices_from(RSM_ctrl, k=-1)]
            flat_ctrl_l.append(flat_ctrls)
        flat_ctrls = np.array(flat_ctrl_l).T

        nan_cols = np.isnan(flat_ctrls).any(axis=0)
        n_nans_cols = np.sum(nan_cols)
        n_bad_cols = n_nans_cols + n_skip_cols
        n_good_cols = len(ROIs_ctrl) - n_bad_cols
        n_ctrls = len(ROIs_ctrl)

        if n_good_cols == 0: # accommodate the sns with missing runs
            print(f'No good columns ({sn})! {n_ctrls=}, {n_nans_cols=}, '
                  f'{n_skip_cols=}')
            good_rows = ~np.isnan(flat_ctrls).any(axis=1)
            print(f'\t{np.sum(good_rows)=}')
            flat_ctrls = flat_ctrls[good_rows, :]
            flat_stim = flat_stim[good_rows]
            nan_cols = np.isnan(flat_ctrls).any(axis=0)
            n_good_cols = np.sum(~nan_cols)
            print(f'\tAfter dropping rows: {n_good_cols=}')
            flat_ctrls = flat_ctrls[:, ~nan_cols]
        elif n_bad_cols > 10:
            print(f'Lots of bad columns ({sn})! {n_ctrls=}, {n_nans_cols=}, '
                  f'{n_skip_cols=}')
            flat_ctrls = flat_ctrls[:, ~nan_cols]
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
            try:
                solution, residuals, rank, s = np.linalg.lstsq(X, IRAFs_focus,
                                                           rcond=None)
            except np.linalg.LinAlgError as e:
                print(f'{sn}: {fp} | {ROI_focus=}, {ROIs_ctrl=} | {e=}')
                return np.nan, np.nan
            IRAFs_focus -= np.dot(np.array(IRAFs_ctrl).T, solution[1:])

            M = np.nanmean(IRAFs_focus)
            return M, (M ** 2) * np.sign(M)

        X = np.hstack([flat_itr[:, None], flat_focus[:, None], flat_ctrls])
    else:
        if regress_row:
            IRAFs_focus = get_IRAFs(RSM_stim, RSM_focus, None,
                                    within_to_nan=RDM_method == 'within_nan',
                                    by_run=False, second_order='corr')
            return IRAFs_focus
        else:
            X = np.hstack([flat_itr[:, None], flat_focus[:, None]])
    if RDM_method == 'within_nan':
        X = X[~np.isnan(flat_focus), :]
        flat_stim = flat_stim[~np.isnan(flat_focus)]

    # if np.sum(np.isnan(X)) > 0:
    #     plt.title(f'X, {sn}')
    #     plt.imshow(X, aspect='auto', interpolation='none')
    #     plt.show()
    #
    #     plt.plot(flat_stim)
    #     plt.show()

    assert np.sum(np.isnan(X)) == 0, f'{sn=}, {ROI_focus=}, {ROIs_ctrl=}, {fp=}'

    solution, residuals, rank, s = np.linalg.lstsq(X, flat_stim, rcond=None)

    p = X.shape[1] - 1
    n = X.shape[0] - 1 # minus 1 because of the intercept
    r_sq = 1 - residuals / np.sum((flat_stim - np.mean(flat_stim)) ** 2)

    try:
        r_sq = r_sq[0]
    except:
        return solution[1], r_sq

    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p - 1)
    return solution[1], r_sq

def do_regr_ISPC_sn(sn, ROI_focus, ROIs_ctrl, fp, trial_similarity,
                   second_order, RDM_method, stdize_by_run, semantic,
                   fp0, fp1, four_tasks, regress_row=False):

    dir_in = fr'cache/conn_RSA/ars/ISPC'
    dir_focus = f'{dir_in}/{fp}_{trial_similarity}'
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus}.npy'
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

def send_to_specific(kwargs, RSA, ISPC=False, ERS_alt=False,
                     easy_override=False):
    fps = prep_fps(kwargs['four_tasks'])
    scores = []
    r_sqs = []
    kwargs['ROIs_ctrl'] = sorted(kwargs['ROIs_ctrl'])  # needed for pkl wrap

    if RSA:
        for fp in fps:
            kwargs['fp'] = fp
            kwargs['cv'] = False
            # print(kwargs['sn'])
            # kwargs['regress_row'] = regress_row
            try:
                score, r_sq = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                          verbose=-1,
                                          easy_override=easy_override)
            except ValueError as e:
                if kwargs["sn"] != '131':
                    print(f'Bad {kwargs["sn"]}: {e}')
                continue
            if not isinstance(r_sq, np.float64):
                # warnings.warn(f'{kwargs["sn"]}, {fp} yields no r_sq: {r_sq=}')
                continue
            scores.append(score)
            r_sqs.append(r_sq)
    elif ISPC:
        for fp in fps:
            kwargs['fp'] = fp
            # kwargs['regress_row'] = regress_row
            try:
                score, r_sq = pickle_wrap(do_regr_ISPC_sn, kwargs=kwargs,
                                          verbose=-1,
                                          easy_override=easy_override)
            except ValueError as e:
                print(f'Bad {kwargs["sn"]}: {e}')
                continue
            if isinstance(score, float) and np.isnan(score):
                continue
            scores.append(score)
            r_sqs.append(r_sq)
    else:
        for fp0 in fps:
            for fp1 in fps:
                if fp0 >= fp1: continue
                kwargs['fp0'] = fp0
                kwargs['fp1'] = fp1
                f = do_RSM_ERS_sn if ERS_alt else do_regr_ERS_sn

                score, r_sq = pickle_wrap(f, kwargs=kwargs, verbose=-1,
                                          easy_override=easy_override)
                if isinstance(score, float) and np.isnan(score):
                    continue
                scores.append(score)
                # print(f'{len(score)=}')
                r_sqs.append(r_sq)

    return np.nanmean(scores), np.nanmean(r_sqs)

def run_all_sn(kwargs, RSA, ISPC, ERS_alt, easy_override=False,
               skip_sns_bonus=None):
    # t_st = time()
    z_l = []
    r_sqs = []
    sns = get_sns('all')['healthy']
    # print(f'{sns=}')
    num_predictors = 1 + len(kwargs['ROIs_ctrl'])
    preds = [kwargs['ROI_focus']] + kwargs['ROIs_ctrl']
    successful_sns = []
    for sn in sns:#, desc='conn regressing...', position=0, leave=False):
        # if sn in ['138', '224']: continue
        if skip_sns_bonus and sn in skip_sns_bonus:
            continue
        kwargs['sn'] = sn
        try:
            # print(f'{sn=}')
            z, r_sq = send_to_specific(kwargs, RSA, ISPC, ERS_alt,
                                       easy_override=easy_override)
        except FileNotFoundError as e:
            # if sn == '102':
            # if '230' in str(e) or '234' in str(e) or '239' in str(e):
            #     continue
            print(f'sn: {e}')
            continue
        if np.isnan(z):
            continue
        # print(f'Did: {sn}')
        z_l.append(z)
        r_sqs.append(r_sq)
        successful_sns.append(sn)
    # print(kwargs)
    schemePE_sns = ['102', '103', '104', '105', '106', '107', '108', '109',
                    '110', '111', '112', '113', '114', '115', '116', '117',
                    '118', '119', '120', '123', '124', '125', '126', '127',
                    '128', '129', '130', '131', '132', '133', '134', '135',
                    '136', '137', '138', '201', '202', '203', '204', '205',
                    '206', '207', '208', '209', '210', '211', '212', '213',
                    '214', '215', '216', '217', '218', '219', '221', '222',
                    '224', '225', '227', '230', '231', '232', '233', '234',
                    '235', '239']
    acceptable_failures = ['116', '125', '133', '213', '215', '231']
    for sn in schemePE_sns:
        if sn in acceptable_failures:
            continue
        if sn not in successful_sns:
            print(f'Not in DistRep: {sn}')
            raise ValueError(f'Failed conn_regress ({sn}): {kwargs=}')

    # print(f'{successful_sns=}')
    # print(f'{schemePE_sns=}')
    # quit()
    t, p = stats.ttest_1samp(z_l, 0)
    M_r_sq = np.nanmean(np.array(r_sqs))
    focus = kwargs['ROI_focus']
    ctrl = kwargs['ROIs_ctrl']
    print(f't[{len(z_l)-1}]={t:.3f}, {p=:.3f} | {M_r_sq=:.5%} '
          f': {focus=}, {ctrl=}')
    assert len(z_l) == 60, f'Bad length ({len(z_l)=}: {kwargs=}'
    # quit()
    return t, np.nanmean(np.array(r_sqs))

def get_title(RSA, ISPC, kwargs, fontsize):
    if 'PFC_ACC' in kwargs['ROI_focus']:
        region = 'Prefrontal'
    elif 'MTL' in kwargs['ROI_focus']:
        region = 'Med. Temp. Lobe'
    elif 'Dorsal' in kwargs['ROI_focus']:
        region = 'Parietal lobe'
    elif 'Ventral' in kwargs['ROI_focus']:
        region = 'Temporal lobe'
    elif 'Subcort' in kwargs['ROI_focus']:
        region = 'Subcortical'
    elif 'Occipital' in kwargs['ROI_focus']:
        region = 'Occipital lobe'
    else:
        region = kwargs['ROI_focus'].split('_')[0]
        # raise ValueError

    regress_row_str = '_regr_row' if kwargs['regress_row'] else ''

    if RSA:
        if kwargs['semantic']:
            if 'Temp' in region:
                fontsize = 23
            else:
                fontsize = 23
            title = f'RSA (semantic): {region}'
            fn = f'RSA_semantic{regress_row_str}_{kwargs["four_tasks"]}.png'
        else:
            if 'Temp' in region:
                fontsize = 23
            else:
                fontsize = 23
            title = f'RSA (perceptual): {region}'
            fn = f'RSA_perceptual{regress_row_str}_{kwargs["four_tasks"]}.png'
    elif ISPC:
        title = f'ISPS: {region}'
        fn = f'ISPS{regress_row_str}_{kwargs["four_tasks"]}.png'
    else:
        title = f'NPS: {region}'
        fn = f'NPS{regress_row_str}_{kwargs["four_tasks"]}.png'
    return title, fn, fontsize


def plot_stacked_bars(kwargs, RSA, ISPC, ERS_alt, easy_override=False,
                      plot=True,
                      ctrl_large_strict=True, # control mega-ROI with separate ROI
                      ctrl_large_w_avg=False,
                      ctrl_large_strict_and_avg=True,

                      ctrl_small_strict=False,

                      ctrl_avg_w_large=False,
                      ctrl_avg_strict=True,

                      voxel_small_M=None, voxel_small_all=None,
                      voxel_large=None, avg_large=None,

                      plot_two=False):
    print(f'{RSA=} ({kwargs["semantic"]}) | {ISPC=}')

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus'] = voxel_small_M
    if ctrl_small_strict:
        kwargs_['ROIs_ctrl'] = [voxel_large]
    else:
        kwargs_['ROIs_ctrl'] = [avg_large]
    t_ROIs_all, r_sqs_ROIs_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
                                            easy_override=easy_override)

    kwargs_['ROIs_ctrl'] = []
    t_ROIs, r_sqs_ROIs = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
                                    easy_override=easy_override)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus'] = voxel_large
    if ctrl_large_strict_and_avg:
        kwargs_['ROIs_ctrl'] = voxel_small_all + [avg_large]
    elif ctrl_large_w_avg:
        kwargs_['ROIs_ctrl'] = [avg_large]
    elif ctrl_large_strict:
        kwargs_['ROIs_ctrl'] = voxel_small_all
    else:
        kwargs_['ROIs_ctrl'] = [voxel_small_M]
    t_large_all, r_sqs_large_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
                                              easy_override=easy_override)
    kwargs_['ROIs_ctrl'] = []
    t_large, r_sqs_large = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
                                      easy_override=easy_override)

    kwargs_ = kwargs.copy()
    kwargs_['ROI_focus'] = avg_large
    if ctrl_avg_strict:
        kwargs_['ROIs_ctrl'] = voxel_small_all
    elif ctrl_avg_w_large:
        kwargs_['ROIs_ctrl'] = [voxel_large]
    else:
        kwargs_['ROIs_ctrl'] = [voxel_small_M]

    t_avg_all, r_sqs_bold_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
                                            easy_override=easy_override)
    kwargs_['ROIs_ctrl'] = []
    t_avg, r_sqs_bold = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
                                    easy_override=easy_override)

    print('-')
    print(f'{t_ROIs=:.3f} ({r_sqs_ROIs:.4f})')
    print(f'\t{t_ROIs_all=:.3f} ({r_sqs_ROIs_all:.4f})')
    print(f'{t_avg=:.3f} ({r_sqs_bold:.4f})')
    print(f'\t{t_avg_all=:.3f} ({r_sqs_bold_all:.4f})')

    t_ROIs_all = max(0, t_ROIs_all)
    t_avg_all = max(0, t_avg_all)
    t_large_all = max(0, t_large_all)

    t_ROIs_dif = t_ROIs - t_ROIs_all
    t_ROIs_dif = max(0, t_ROIs_dif)
    t_bold_dif = t_avg - t_avg_all
    t_bold_dif = max(0, t_bold_dif)
    t_large_dif = t_large - t_large_all
    t_large_dif = max(0, t_large_dif)

    title, fn, fontsize = get_title(RSA, ISPC, kwargs_, 28)
    if plot:
        plt.rcParams.update({'font.size': 20})
        if plot_two:
            bar_names = ['Small\nvoxel', 'Large\naverage', ]
            colors = ['dodgerblue', 'crimson',]
            plt.bar(bar_names,
                    [t_ROIs_dif, t_bold_dif],
                    bottom=[t_ROIs_all, t_avg_all],
                    label='2', color='darkgray',
                    linewidth=1., edgecolor='k')
            plt.bar(bar_names,
                    [t_ROIs_all, t_avg_all],
                    bottom=[0, 0], label='1',
                    color=colors,
                    linewidth=1., edgecolor='k')
            max_height = max(t_ROIs, t_avg, t_ROIs_all, t_avg_all, 4.)
        else:
            bar_names = ['Small\nvoxel', 'Large\nvoxel', 'Large\naverage', ]
            colors = ['dodgerblue', 'orange', 'crimson',]
            plt.bar(bar_names,
                    [t_ROIs_dif, t_large_dif, t_bold_dif],
                    bottom=[t_ROIs_all, t_large_all, t_avg_all],
                    label='2', color='darkgray',
                    linewidth=1., edgecolor='k')
            plt.bar(bar_names,
                    [t_ROIs_all, t_large_all, t_avg_all],
                    bottom=[0, 0, 0], label='1',
                    color=colors,
                    linewidth=1., edgecolor='k')
            max_height = max(t_ROIs, t_avg, t_large, t_ROIs_all, t_avg_all,
                             t_large_all, 4.)

        plt.xticks(bar_names, fontsize=23)
        plt.ylabel('t-value', fontsize=28)

        print(f'{title=}')
        fn = f'{kwargs_["ROI_focus"]}_{fn}'

        plt.gca().spines[['top', 'right']].set_visible(False)

        if np.isnan(max_height):
            print('NaN max_height!')
            max_height = 0
        plt.yticks(range(0, int(max_height*1.07) + 1,
                         min(max(int(max_height*1.07) // 4, 1), 5)),
                   fontsize=26)
        plt.ylim(0, max_height * 1.07)
        plt.title(title, fontsize=fontsize, pad=15)
        plt.tight_layout()
        if 'RSA' in fn:
            fp_out = rf'connRSA/stacked_bar/RSA/{fn}'
            Path(fp_out).parent.mkdir(exist_ok=True, parents=True)
        elif 'ISPC' in fn:
            fp_out = rf'connRSA/stacked_bar/ISPC/{fn}'
            Path(fp_out).parent.mkdir(exist_ok=True, parents=True)
        else:
            fp_out = rf'connRSA/stacked_bar/ERS/{fn}'
            Path(fp_out).parent.mkdir(exist_ok=True, parents=True)

        plt.savefig(fp_out, dpi=300)
        plt.show()

    return t_ROIs_all, t_avg_all, t_large_all, title


def prep_ROI_avg(target_name, ROI_cols,
                 fp0, fp1, trial_similarity, stdize_by_run, semantic,
                 second_order, RDM_method, fp,
                 four_tasks='8',
                 RSA=False, ISPC=False,
                 ERS_alt=False, regress_row=False):
    kwargs = locals().copy()
    if RSA or ISPC or ERS_alt:
        if fp is None:
            fps = prep_fps(four_tasks)
            for fp in fps:
                kwargs['fp'] = fp
                prep_ROI_avg(**kwargs)
            return
    else:
        if fp0 is None:
            fps = prep_fps(four_tasks)
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
    for sn in sns:
        fp_out = f'{dir_focus}/{sn}_{target_name}.npy'
        if os.path.isfile(fp_out) and os.path.getsize(fp_out) > 2048:
            continue

        skips = 0
        l = []
        for ROI_ctrl in ROI_cols:
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                skips += 1
                continue
            try:
                with open(fp_ctrl, 'rb') as f:
                    ISPC_ctrl = np.load(f)
                    l.append(ISPC_ctrl)
            except ValueError as e:
                print(f'Bad {sn} ({ROI_ctrl}): {e}\n\t{fp_ctrl=}')

        if len(l) == 0:
            continue

        ctrl_M = np.nanmean(np.array(l), axis=0)
        with open(fp_out, 'wb') as f:
            np.save(f, ctrl_M)


def do_regr():
    easy_override = False
    ctrl_strict = False

    ISPC = False
    RSA = False
    semantic = False
    ERS_alt = False
    conn = 'prod'
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    regress_row = False
    stdize_by_run = True if trial_similarity == 'euc' else False
    # stdize_by_run = False
    # TODO: update ISPC to have stdize_by_run as a toggle
    # target_ROI = 'Occipital'
    # target_ROI = 'FP'
    target_ROI = 'Hipp'
    target_ROI = 'SFG'
    # target_ROI = 'MTL2'
    # atlas = get_atlas()
    # print(atlas['ROIs'])
    # TODO: loop over every anatomical region and plot system vs ROI bias

    # ['Hipp', 'ATL', 'PhG', 'STG', 'MTG', 'ITG']

    target_ROIs = ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'ATL', 'STG',
                   'MTG', 'ITG', 'FuG', 'PhG', 'pSTS', 'SPL', 'IPL', 'Pcun',
                   'PoG', 'INS', 'PCC', 'ACC', 'EVC', 'LOC', 'sOcG', 'Amyg',
                   'Hipp', 'Str', 'Tha']
    # print(len(target_ROIs))
    # quit()

    # target_ROIs = ['Occipital', 'Ventral', 'Dorsal']

    l = [(False, False, False, False),
         (True, False, False, False),
         (False, True, False, False),
         (False, True, True, False),]

    target_ROIs = ['Occipital', 'Ventral', 'Dorsal', 'PFC', #'cingulate',
                   'subcort']
    # target_ROIs = ['Ventral',]


    # for (ISPC, RSA, semantic, ERS_alt) in l:
    #     if ISPC: continue
    #     if not RSA: continue
    #     if not semantic: continue
        # ts_ROI, ts_BOLD, ts_conn = [], [], []
        # if not semantic: continue
        # if not RSA or ISPC: continue
    ts_ROI, ts_BOLD, ts_conn = [], [], []
    for regress_row in [False]:
        for target_ROI in target_ROIs: # Occipital', 'MTL', 'PFC_ACC
            # ROI_focus = f'{target_ROI}_{conn}'
            # ROI_focus = f'{target_ROI}_BOLD_cmb'

            regions = set(prep_networks(
                network_setting=ROI2NETWORK[target_ROI])[target_ROI])
            ROIs_match = [ROI for region in regions
                              for ROI in get_BNA_ROIs() if region in ROI]
            ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]

            kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
                      'fp1': None, 'trial_similarity': trial_similarity,
                      'second_order': second_order,
                      'RDM_method': RDM_method,
                      'stdize_by_run': stdize_by_run,
                      'regress_row': regress_row, 'four_tasks': four_tasks,
                      }

            target_name = fr'{target_ROI}_M'
            prep_ROI_avg(target_name, ROI_lvl_control, RSA=RSA, ISPC=ISPC,
                         ERS_alt=ERS_alt, **kwargs)

            outer_kwargs = {'kwargs': kwargs, 'RSA': RSA, 'ISPC': ISPC,
                           'ERS_alt': ERS_alt,
                           'easy_override': easy_override,
                           'plot': len(target_ROIs) < 10,
                           'ctrl_large_strict': ctrl_strict,
                           'ctrl_avg_strict': ctrl_strict}

            outer_kwargs['voxel_small_M'] = target_name
            outer_kwargs['voxel_small_all'] = ROI_lvl_control
            outer_kwargs['voxel_large'] = f'{target_ROI}_BOLD_cmb'
            outer_kwargs['avg_large'] = f'{target_ROI}_BOLD'

            t_ROIs_all, t_bold_all, t_conn_all, title = (
                pickle_wrap(plot_stacked_bars, kwargs=outer_kwargs,
                            verbose=-1, easy_override=easy_override))

            print(f'{target_ROI} | {t_ROIs_all:.2f}, {t_bold_all:.2f} '
                  f'{t_conn_all:.2f}')

            ts_ROI.append(t_ROIs_all)
            ts_BOLD.append(t_bold_all)
            ts_conn.append(t_conn_all)
    # print(f'{len(t_ROIs_all)=}')
    # print(f'{ts_ROI=}')
    if len(target_ROIs) >= 10:
        if len(target_ROIs) > 40:
            cbl = False
        else:
            cbl = True
        # print(len(target_ROIs))
        atlas = get_atlas(combine_regions=True, combine_bilateral=cbl)
        assert len(atlas['tick_labels']) == len(target_ROIs)
        # print(atlas['tick_labels'])
        # print(ts_conn)
        # quit()

        # vmax = max(max(ts_ROI), max(ts_BOLD), max(ts_conn))
        vmax = 4
        thresh = 1.65

        print(f'{ts_ROI=}')
        title_ROI = title.split(':')[0] + ': ROIs'
        my_plot_surf(ts_ROI, atlas, title_ROI, vmax=vmax, thresh=thresh)

        title_system = title.split(':')[0] + ': System'
        my_plot_surf(ts_BOLD, atlas, title_system, vmax=vmax, thresh=thresh)

        title_conn = title.split(':')[0] + ': Large'
        my_plot_surf(ts_conn, atlas, title_conn, vmax=vmax, thresh=thresh)

def plot_basic():
    ISPC = False
    RSA = False
    semantic = False
    ERS_alt = False
    conn = 'prod'
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    regress_row = False
    stdize_by_run = True if trial_similarity == 'euc' else False
    # stdize_by_run = False
    # TODO: update ISPC to have stdize_by_run as a toggle
    # target_ROI = 'Occipital'
    # target_ROI = 'FP'
    target_ROI = 'Hipp'
    target_ROI = 'SFG'
    # target_ROI = 'MTL2'



    # TODO: loop over every anatomical region and plot system vs ROI bias

    # ['Hipp', 'ATL', 'PhG', 'STG', 'MTG', 'ITG']

    target_ROIs = ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'ATL', 'STG',
                   'MTG', 'ITG', 'FuG', 'PhG', 'pSTS', 'SPL', 'IPL', 'Pcun',
                   'PoG', 'INS', 'PCC', 'ACC', 'EVC', 'LOC', 'sOcG', 'Amyg',
                   'Hipp', 'Str', 'Tha']


    l = [(False, False, False, False),
         (True, False, False, False),
         (False, True, False, False),
         (False, True, True, False),]

    target_ROIs = ['Occipital', 'Ventral', 'Dorsal', 'PFC', #'cingulate',
                   'subcort']
    # target_ROIs = ['Ventral',]

    ROI_foci = [f'{ROI}_BOLD' for ROI in get_BNA_ROIs()]

    for (ISPC, RSA, semantic, ERS_alt) in l:
        if ISPC: continue
        if not RSA: continue
        if not semantic: continue
        # ts_ROI, ts_BOLD, ts_conn = [], [], []
        # if not semantic: continue
        # if not RSA or ISPC: continue
        ts_ROI, ts_BOLD, ts_conn = [], [], []
        kwargs_ = None
        for ROI_focus in ROI_foci:
            kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
                      'fp1': None, 'trial_similarity': trial_similarity,
                      'second_order': second_order,
                      'RDM_method': RDM_method,
                      'stdize_by_run': stdize_by_run,
                      'regress_row': regress_row, 'four_tasks': four_tasks,
                      }

            kwargs_ = kwargs.copy()
            kwargs_['ROI_focus'] = ROI_focus
            kwargs_['ROIs_ctrl'] = []
            t_ROIs_all, r_sqs_ROIs_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
                                                    easy_override=False,
                                                    skip_sns_bonus=('131',
                                                                    '132'))
            # print(t_ROIs_all)
            ts_ROI.append(t_ROIs_all)

        title, fn, fontsize = get_title(RSA, ISPC, kwargs_, 28)
        print(f'{ts_ROI=}')
        vmax = 4
        thresh = 1.65
        title_ROI = title.split(':')[0] + ': ROIs'
        atlas = get_atlas()
        my_plot_surf(ts_ROI, atlas, title_ROI, vmax=vmax, thresh=thresh)


import sys
sys.setrecursionlimit(10000)

if __name__ == '__main__':
    do_regr()
    # plot_basic()

