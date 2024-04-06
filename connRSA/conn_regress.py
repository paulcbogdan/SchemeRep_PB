import os

import pandas as pd

from connRSA.conn_analyze_IRAFs import prep_network2ROI, ROI2NETWORK
from connRSA.conn_utils import get_BNA_ROIs
from fMRI_proc import within_run_to_nan
from old.networks import prep_networks

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

import numpy as np
import scipy.stats as stats

from time import time
from org_sns import get_sns
import matplotlib.pyplot as plt

def do_regr_ERS_sn(sn, ROI_focus_conn, ROIs_ctrl,
                   fp0, fp1, trial_similarity, stdize_by_run):
    dir_focus = fr'cache/conn_RSA/ars/ERS'
    dir_focus = fr'{dir_focus}/{fp0}_{fp1}_{trial_similarity}_{stdize_by_run}'
    fp_focus = f'{dir_focus}/{sn}_{ROI_focus_conn}.npy'
    with open(fp_focus, 'rb') as f:
        ERS_focus = np.load(f)
    flat_focus = ERS_focus.flatten()

    ERS_ctrl_l = []
    for ROI_ctrl in ROIs_ctrl:
        fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
        with open(fp_ctrl, 'rb') as f:
            ERS_ctrl = np.load(f)
        ERS_ctrl_l.append(ERS_ctrl)
    ERS_ctrl = np.array(ERS_ctrl_l)
    flat_ctrl = ERS_ctrl.reshape(ERS_ctrl.shape[0], -1).T
    solution, residuals, rank, s = np.linalg.lstsq(flat_ctrl, flat_focus,
                                                   rcond=None)
    ERS_focus -= np.dot(flat_ctrl, solution) # TODO: double-check

    ERS_sames = np.diag(ERS_focus)
    ERS_focus_ = ERS_focus.copy()
    ERS_focus_[np.eye(len(ERS_focus_), dtype=bool)] = np.nan
    ERS_elses = np.nanmean(ERS_focus_, axis=1)
    ERS_dif = ERS_sames - ERS_elses
    score = np.nanmean(ERS_dif)

    return score


def do_regr_RSA_sn(sn, ROI_focus_conn, ROIs_ctrl, fp, trial_similarity,
                   second_order, RDM_method, stdize_by_run, semantic):

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
        for ROI_ctrl in ROIs_ctrl:
            # conn = ROI_ctrl.split('_')[-1]
            # dir_focus = (f'{dir_in}/{fp}_{trial_similarity}_'
            #              f'{second_order}_{RDM_method}_{stdize_by_run}')
            # ROI_ctrl_ = '_'.join(ROI_ctrl.split('_')[:-1])
            fp_ctrl = f'{dir_focus}/{sn}_{ROI_ctrl}.npy'
            with open(fp_ctrl, 'rb') as f:
                RSM_ctrl = np.load(f)
            flat_ctrl = RSM_ctrl[np.tril_indices_from(RSM_ctrl, k=-1)]
            flat_ctrls.append(flat_ctrl)
        flat_ctrl = np.array(flat_ctrls).T
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



def do_regr():
    RSA = True
    semantic = False
    conn = 'BOLD'
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    four_tasks = '8'
    combine_regions = False
    split = False
    RDM_method = 'within_nan'
    age = 'healthy'
    stdize_by_run = True if trial_similarity == 'euc' else False
    target_ROI = 'Occipital'

    sns = get_sns()['healthy']
    ROI_focus = f'{target_ROI}_{conn}'

    regions = set(prep_networks(
        network_setting=ROI2NETWORK[target_ROI])[target_ROI])
    ROIs_match = [ROI for region in regions
                      for ROI in get_BNA_ROIs() if region in ROI]
    ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]
    ROIs_ctrl = ROI_lvl_control

    # ROIs_ctrl = [f'{target_ROI}_BOLD']
    # ROIs_ctrl = []
    # sn = sns[0]
    settings = {'ROI_focus': ROI_focus, 'ROIs_ctrl': ROIs_ctrl,
                'fp': f'obj8_fMRI',
                'trial_similarity': trial_similarity,
                'second_order': second_order, 'RDM_method': RDM_method,
                'stdize_by_run': stdize_by_run, 'semantic': semantic}
    t_st = time()
    z_l = []
    for sn in sns:
        settings['sn'] = sn
        try:
            z = do_regr_RSA_sn(**settings)
        except FileNotFoundError:
            pass
        # print(f'{z=:.3f}')
        z_l.append(z)
    print(f'{time() - t_st:.2f} s')
    t, p = stats.ttest_1samp(z_l, 0)
    print(f'{t=:.3f}, {p=:.3f}')


if __name__ == '__main__':
    do_regr()











