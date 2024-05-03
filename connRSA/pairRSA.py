import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from tqdm import tqdm

from atlas_utils import get_atlas
from connRSA.single_trial_conn import prep_vecs, prep_fps
from fMRI_proc import within_run_to_nan
from old.modularity import get_binary_matrix, get_modules, get_partition_matrix
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from stim import get_stim_RDM
from utils import pickle_wrap

import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')

def pairRSA(fp, within2nan=True, semantic=True, regress_global=True):
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              'combine_regions': False,
              'combine_bilateral': False,
              'get_df_sn': True
              }

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')
    sn_act = np.nanmean(sn_inc_activity, axis=1)
    if regress_global:
        sn_act_M = np.nanmean(sn_act, axis=1) # add global signal as ROI
        sn_act = np.concatenate([sn_act, sn_act_M[:, None]], axis=1)

    d_vecs = prep_vecs(True, semantic)
    RSM_RSM_l = []
    sns = []
    for sn_i, df_sn in tqdm(enumerate(df_sns_l), desc='pairRSA by sn'):
        sn = df_sn['sn'].iloc[0]
        act = sn_act[sn_i]
        act_std = stats.zscore(act, axis=-1)
        act_std0 = act_std[:, None, :]
        act_std1 = act_std[None, :, :]

        binary_corr = act_std0 > act_std1
        binary_corr0 = binary_corr[..., None]
        binary_corr1 = binary_corr[..., None, :]

        # NaN becomes True if type is left as bool
        binary_corr_RSM = (binary_corr0 == binary_corr1).astype(np.float32)

        tril_idxs = np.tril_indices(114, k=-1)
        if within2nan:
            trial_per_run = binary_corr_RSM.shape[-1] // 3
            for run in range(3):
                low = run * trial_per_run
                high = (run + 1) * trial_per_run
                binary_corr_RSM[..., low:high, low:high] = np.nan

        binary_corr_RSM = binary_corr_RSM[..., *tril_idxs]
        binary_corr_RSM[np.diag_indices(binary_corr_RSM.shape[0])] = np.nan
        binary_corr_RSM = stats.zscore(binary_corr_RSM, axis=-1,
                                       nan_policy='omit')

        RSM_stim = pickle_wrap(get_stim_RDM, None,
                               kwargs={'df_sn': df_sn, 'd_vecs': d_vecs,
                                       'obj_only': True,}, verbose=-1)

        RSM_stim = within_run_to_nan(RSM_stim)
        RSM_stim[np.diag_indices_from(RSM_stim)] = np.nan

        RSM_stim = RSM_stim[tril_idxs]
        RSM_stim = stats.zscore(RSM_stim, nan_policy='omit')

        if regress_global:
            global_row = binary_corr_RSM[:-1, -1, :]
            non_global_RSM = binary_corr_RSM[:-1, :-1, :]
            mat_RSM_RSM = []
            num_nanner_rows = 0
            for j, row in enumerate(non_global_RSM):
                regressors = np.array([RSM_stim, global_row[j]]).T
                nans = np.any(np.isnan(regressors), axis=1)
                n_non_nans = np.sum(~nans)
                if n_non_nans < 2:
                    # print('all nan??')
                    num_nanner_rows += 1
                    mat_RSM_RSM.append(np.full(row.shape[0], np.nan))
                    continue
                regressors = regressors[~nans, :]
                XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
                XTX_invX = np.dot(XTX_inv, regressors.T)
                betas = np.dot(XTX_invX, row.T[~nans, :])
                mat_RSM_RSM.append(betas[0, :])
            print(f'{num_nanner_rows=}')
        else:
            mat_RSM_RSM = np.nanmean(binary_corr_RSM * RSM_stim[None, None],
                                     axis=-1)
        RSM_RSM_l.append(mat_RSM_RSM)
        sns.append(sn)

    return RSM_RSM_l, sns

def run_pairRSA(fp, within2nan=True, semantic=False, regress_global=False):
    # for key in ['7', '8']:
    sns_use = set()
    RSM_RSM_l_l = []
    sns_l = []
    for i, fp in enumerate(prep_fps('7')):
        RSM_RSM_l, sns = pickle_wrap(pairRSA, None,
                                kwargs={'fp': fp, 'within2nan': within2nan,
                                        'semantic': semantic,
                                        'regress_global': regress_global},
                                verbose=1, easy_override=False)
        if i == 0:
            sns_use = set(sns)
        else:
            sns_use = sns_use.intersection(sns)
        RSM_RSM_l_l.append(RSM_RSM_l)
        sns_l.append(sns)
    RSM_RSM_l_l_ = []
    for RSM_RSM_l, sns in zip(RSM_RSM_l_l, sns_l):
        idxs = [i for i, sn in enumerate(sns) if sn in sns_use]
        RSM_RSM_l_l_.append(np.array(RSM_RSM_l)[idxs])
        # print(idxs)
        # quit()
        # sns_use.update(sns)
        # print(f'{len(sns_use)=}')
    # quit()
    # print(np.array(RSM_RSM_l).shape)
    # quit()
    # RSM_RSM_l_l = np.array(RSM_RSM_l_l_)
    RSM_RSM_l = np.nanmean(np.array(RSM_RSM_l_l_), axis=0)
    # print(RSM_RSM_l_l.shape)
    # quit()

    RSM_RSM_l = np.array(RSM_RSM_l)
    # print(RSM_RSM_l)
    # quit()
    RSM_RSM_M = np.nanmean(RSM_RSM_l, axis=0)
    RSM_RSM_SE = stats.sem(RSM_RSM_l, axis=0, nan_policy='omit')
    RSM_RSM_t = RSM_RSM_M / RSM_RSM_SE
    RSM_RSM_z = stats.norm.ppf(stats.t.cdf(RSM_RSM_t, RSM_RSM_l.shape[0] - 1))

    # for i, row in
    RSM_RSM_bin, _ = get_binary_matrix(RSM_RSM_z, threshold=.8, intersect=True)

    partitions, matrix_mask = \
        get_main_partitions(z_both, coords=atlas['coords'], plot=plot,
                            threshold=thr,
                            fn_str='', overlapping=False,
                            dir_out_full=dir_out,
                            title_extra=title_extra,
                            )

    partitions = get_modules(RSM_RSM_bin)
    atlas = get_atlas()

    for p in partitions:
        p_mat = get_partition_matrix(RSM_RSM_bin, p, w_zeros=True)
        p_mat[p_mat < .5] = np.nan
        plot_connectivity(p_mat, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'],
                          title='pairRSA', no_avg=True, cbar_label='t-value',
                          vmin=-4, vmax=4)
    quit()

    # print(RSM_RSM_z.shape)
    # quit()
    #
    plot_connectivity(RSM_RSM_z, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title='pairRSA', no_avg=True, cbar_label='t-value',
                      vmin=-4, vmax=4)

if __name__ == '__main__':
    # run_pairRSA()
    run_pairRSA('obj7_fMRI') # TODO: regress_global = False