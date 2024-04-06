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
        try:
            solution, residuals, rank, s = np.linalg.lstsq(flat_ctrl, flat_focus,
                                                           rcond=None)
        except np.linalg.LinAlgError as e:
            print(f'{e=}')
            print(f'{flat_focus=}')
            plt.imshow(flat_ctrl, aspect='auto')
            plt.show()
            quit()
        ERS_ctrl = np.transpose(ERS_ctrl, (1, 2, 0))
        ERS_focus -= np.dot(ERS_ctrl, solution) # TODO: double-check

    ERS_sames = np.diag(ERS_focus)
    ERS_focus_ = ERS_focus.copy()
    ERS_focus_[np.eye(len(ERS_focus_), dtype=bool)] = np.nan
    ERS_elses = np.nanmean(ERS_focus_, axis=1)
    ERS_dif = ERS_sames - ERS_elses
    score = np.nanmean(ERS_dif)

    return score


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
    # print(f'{flat_stim.shape=}')

    flat_itr = np.array([1] * len(flat_focus))

    if len(ROIs_ctrl):
        flat_ctrls = []
        skips = 0
        for ROI_ctrl in ROIs_ctrl:
            # conn = ROI_ctrl.split('_')[-1]
            # dir_focus = (f'{dir_in}/{fp}_{trial_similarity}_'
            #              f'{second_order}_{RDM_method}_{stdize_by_run}')
            # ROI_ctrl_ = '_'.join(ROI_ctrl.split('_')[:-1])
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            if not os.path.isfile(fp_ctrl):
                skips += 1
                continue
            with open(fp_ctrl, 'rb') as f:
                RSM_ctrl = np.load(f)
            flat_ctrl = RSM_ctrl[np.tril_indices_from(RSM_ctrl, k=-1)]
            flat_ctrls.append(flat_ctrl)
        flat_ctrl = np.array(flat_ctrls).T
        # print(flat_ctrl.shape)
        # quit()
        # flat_ctrl = ERS_ctrl.reshape(ERS_ctrl.shape[0], -1).T # (12996, 22)
        nan_cols = np.isnan(flat_ctrl).any(axis=0)
        n_nan_cols = np.sum(nan_cols) + skips
        # print(f'{n_nan_cols=}')
        # # print(f'Num NaNs: {n_nan_cols=}')
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
    X[:, 1:] = stats.zscore(X[:, 1:], axis=0)
    flat_stim = stats.zscore(flat_stim)
    solution, residuals, rank, s = np.linalg.lstsq(X, flat_stim, rcond=None)
    return solution[1]

    # print(f'{X.shape=}')
    # print(f'{solution=}')
    # compare to statsmodels
    # import statsmodels.formula.api as smf
    # import pandas as pd
    # df = pd.DataFrame({'stim': flat_stim,
    #                    'focus': flat_focus,
    #                    'ctrl': flat_ctrl[:, 0]})
    # df['focus'] = stats.zscore(df['focus'])
    # df['ctrl'] = stats.zscore(df['ctrl'])
    # mod = smf.ols(formula='stim ~ 1 + focus + ctrl', data=df)
    # res = mod.fit()
    # print(res.summary())

def send_to_specific(kwargs, RSA, ISPC=False):
    fps = prep_fps('8')
    scores = []
    if RSA:
        for fp in fps:
            kwargs['fp'] = fp
            score = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                verbose=-1,
                                easy_override=True)
            # score = do_regr_RSA_sn(**kwargs)
            scores.append(score)
        return np.nanmean(scores)
    elif ISPC:
        # for fp0 in fps:
        #     for fp1 in fps:
        #         if fp0 >= fp1: continue
        #         kwargs['fp0'] = fp0
        #         kwargs['fp1'] = fp1
        #         score = do_regr_ISPC_sn(**kwargs)
        #         scores.append(score)
        pass
    else:
        for fp0 in fps:
            for fp1 in fps:
                if fp0 >= fp1: continue
                kwargs['fp0'] = fp0
                kwargs['fp1'] = fp1
                # score = do_regr_ERS_sn(**kwargs)
                score = pickle_wrap(do_regr_ERS_sn, kwargs=kwargs,
                                    verbose=-1)

                scores.append(score)
        return np.nanmean(scores)


def do_regr():
    RSA = True
    semantic = True
    conn = 'BOLD'
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    stdize_by_run = True if trial_similarity == 'euc' else False
    target_ROI = 'MTL'

    sns = get_sns('all')['healthy']
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
    t_st = time()
    z_l = []
    print(f'Num controls: {len(ROIs_ctrl)}')
    for sn in tqdm(sns, desc='conn regressing...', position=0, leave=True):
        if sn in ['138', '224']: continue
        kwargs['sn'] = sn
        try:
            z = send_to_specific(kwargs, RSA)
        except FileNotFoundError as e:
            print(f'sn: {e}')
            continue
        if np.isnan(z):
            continue
        z_l.append(z)
        # print(f'{z=}')
    print(f'{time() - t_st:.2f} s')
    t, p = stats.ttest_1samp(z_l, 0)
    print(f't[{len(z_l)-1}]={t:.3f}, {p=:.3f}')


if __name__ == '__main__':
    do_regr()











