import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
from scipy import stats
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.single_trial_conn import prep_vecs, prep_fps
from old.plot_gen import plot_connectivity
from org_sns import get_sns
from organize_bhv import get_trial_info
from stim import get_semantic_vectors, get_stim_RDM
from utils import pickle_wrap
from functools import cache
from sklearn import decomposition

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
                    second_order, semantic, regress_global=True):
    dir_in = fr'cache/conn_RSA/ars/RSA'
    RDM_method_ = 'within_nan'
    dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
                 f'{second_order}_{RDM_method_}_{stdize_by_run}')
    fp_focus1 = f'{dir_focus1}/{sn}_{ROI}_BOLD.npy'
    with open(fp_focus1, 'rb') as f:
        RSM_focus1 = np.load(f)

    tril_idxs = np.tril_indices(RSM_focus1.shape[0], k=-1)
    RSM_fMRI_flat = RSM_focus1[tril_idxs]
    RSM_fMRI_flat = stats.zscore(RSM_fMRI_flat)

    RSMs = prep_w2v_feature_RSMs(sn, fp, semantic)
    RSM_models_flat = RSMs[:, *tril_idxs]
    RSM_models_flat = stats.zscore(RSM_models_flat, axis=1)

    if regress_global:
        kw = {'sn': sn, 'fp': fp, 'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run, 'second_order': second_order}
        RSM_global = pickle_wrap(get_M_RSM, kwargs=kw)
        RSM_global_flat = RSM_global[tril_idxs]
        regressors = np.array([RSM_fMRI_flat, RSM_global_flat]).T
        nans = np.any(np.isnan(regressors), axis=1)
        regressors = regressors[~nans, :]
        XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
        XTX_invX = np.dot(XTX_inv, regressors.T)
        betas = np.dot(XTX_invX, RSM_models_flat.T)
        RSM_corr = betas[0, :]
    else:
        RSM_corr = np.nanmean(RSM_fMRI_flat[None,  :] *
                              RSM_models_flat, axis=1)
    # print(betas.shape)
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

def run_var_analysis():
    semantic = False
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    second_order = 'spear'

    # sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
    #        '111', '112', '113', '114', '115', '116', '117', '118', '119',
    #        '120', '123', '124', '125', '126', '127', '128', '129', '130',
    #        '131', '132', '133', '134', '135', '136', '137', '138', '201',
    #        '202', '203', '204', '205', '206', '207', '208', '209', '210',
    #        '211', '212', '213', '214', '215', '216', '217', '218', '219',
    #        '221', '222', '224', '225', '227', '230', '231', '232', '233',
    #        '234', '235', '239']
    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117', '118', '119', '120',
           '123', '124', '126', '127', '128', '129', '130', '134', '135',
           '136', '137', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214', '216', '217', '218',
           '219', '221', '222', '225', '227', '232', '233', '235']
    # sns = sns[:2]

    four_tasks = '7'

    atlas = get_atlas()
    ROIs = atlas['ROIs']
    kwargs = {'sns': sns, 'four_tasks': four_tasks,
              'trial_similarity': trial_similarity, 'stdize_by_run': stdize_by_run,
              'semantic': semantic, 'second_order': second_order}
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

    # print(sn_all_vals.shape)
    # quit()
    # plt.hist(np.nanmean(sn_all_vals, axis=(0, 2)), bins=20)
    # plt.show()
    # quit()

    sn_all_vals_conn = np.nanmean((sn_all_vals[..., None, :] *
                                   sn_all_vals[..., None, :, :]), axis=-1)
    print(sn_all_vals_conn.shape)
    print('Prepped')
    M = np.nanmean(sn_all_vals_conn, axis=0)
    print(M)
    SE = stats.sem(sn_all_vals_conn, axis=0, nan_policy='omit')
    N = np.sum(~np.isnan(sn_all_vals_conn), axis=0)
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

# def regress_out_RSMs(sn, ROI0, fp, trial_similarity, stdize_by_run,
#                      second_order):
#     dir_in = fr'cache/conn_RSA/ars/RSA'
#     RDM_method_ = 'within_nan'
#     dir_focus1 = (f'{dir_in}/{fp}_{trial_similarity}_'
#                   f'{second_order}_{RDM_method_}_{stdize_by_run}')
#     fp_focus1 = f'{dir_focus1}/{sn}_{ROI0}_BOLD.npy'
#     with open(fp_focus1, 'rb') as f:
#         RSM_focus1 = np.load(f)
#     RSM_flat0 = RSM_focus1[np.tril_indices(RSM_focus1.shape[0], k=-1)]
#
#     atlas = get_atlas()
#     for ROI1 in atlas['ROIs']:
#         fp_focus2 = f'{dir_focus1}/{sn}_{ROI1}_BOLD.npy'
#         with open(fp_focus2, 'rb') as f:
#             RSM_focus2 = np.load(f)
#         RSM_flat1 = RSM_focus2[np.tril_indices(RSM_focus2.shape[0], k=-1)]
#



if __name__ == '__main__':
    run_var_analysis()
