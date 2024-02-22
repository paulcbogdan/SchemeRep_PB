from collections import defaultdict
from pathlib import Path
import os

import shutil
from nilearn import image

from atlas_utils import get_atlas
from corr_RSA_x_vendor import get_plain_df_sn
from old.modularity import get_partition_cross
from old.plot_gen import plot_connectivity
from org_sns import get_sns
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiMapsMasker
from nilearn.image import high_variance_confounds
import pandas as pd
from time import time

from utils import pickle_wrap, stdize
from ven_x_dor import get_module_cross_trialwise_z
from vendor_partitioning import get_vendor_partitions
import scipy.stats as stats
from scipy import linalg
from random import random

def get_sn_rs(sn):
    fp_in = fr'fMRI_in/{sn}/resting/rs.nii.gz'
    img = image.load_img(fp_in)
    confounds = pd.DataFrame(high_variance_confounds(img, percentile=1))
    img = image.clean_img(img, confounds=confounds)
    data = img.get_fdata()
    return data

def load_resting_data():
    age2sn = get_sns()
    sns = age2sn[1] + age2sn[2]
    atlas = get_atlas()
    sn_roi_act = []
    bad_rs_sns = {'133'}
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    for sn in tqdm(sns, desc='Loading fMRI'):
        # if sn in bad_rs_sns:
        #     continue
        data = pickle_wrap(get_sn_rs, kwargs={'sn': sn})
        ROIs = atlas['ROIs']
        ROI_nums = atlas['ROI_nums']
        ROI_regions = atlas['ROI_regions']
        ar = []
        for j, (ROI, ROI_num, region) in enumerate(
                zip(ROIs, ROI_nums, ROI_regions)):
            atlas_roi = atlas['maps'].get_fdata() == ROI_num
            region_vecs = data[atlas_roi]
            ts = np.nanmean(region_vecs, axis=0)
            ar.append(ts)
        ar = np.array(ar)
        print(f'{sn} | {ar.shape=}')
        sn_roi_act.append(ar)
        # corr = np.corrcoef(ar)
        #
        # plot_connectivity(corr,
        #                   atlas['ticks'],
        #                   atlas['tick_labels'],
        #                   atlas['tick_lows'],
        #                   no_avg=True,
        #                   cbar_label='t-value')
        #
        #
        #
        # quit()
    sn_roi_act = np.array(sn_roi_act)
    return sn_roi_act, sns

def high_variance_conn_confounds(conn_trials, tile=.02, n_confounds=5):
    for i in tqdm(range(conn_trials.shape[0]), desc='high variance confounds'):
        # if i != 27: continue
        sn_conn = conn_trials[i]
        # print(np.sum(np.isnan(sn_conn)))
        trils = np.tril_indices(sn_conn.shape[1], k=-1)
        sn_flat_T = sn_conn[trils[0], trils[1], :].T
        v_flat = np.nanvar(sn_flat_T, axis=0) # participant 27 has all zero in two trials
        v_flat[np.isnan(v_flat)] = 0
        # print(v_flat.shape)
        # plt.hist(v_flat)
        # plt.show()
        # continue
        top_tile_idx = int(v_flat.shape[0] * tile)
        top_idxs = np.argsort(v_flat)[-top_tile_idx:]
        sn_top_T = sn_flat_T[:, top_idxs]
        # print(v_flat[top_idxs])
        # quit()
        num_nans = np.sum(np.isnan(sn_top_T))
        # print(f'{i}: {num_nans}')
        # print(sn_top_T.shape)

        try:
            U, S, Vh = np.linalg.svd(sn_top_T)
        except np.linalg.LinAlgError:
            print(f'LinAlgError: {i}')
            plt.imshow(np.isnan(sn_top_T), aspect='auto')
            plt.colorbar()
            plt.title(f'{num_nans=}')
            plt.show()
            continue
        confounds = U[:, :n_confounds]

        # Taken from nilearn signal.clean
        #   https://pages.stat.wisc.edu/~larget/math496/qr.html
        # I believe:
        #   betas = Q.T.dot(sn_top_T)
        #   so subtracting Q.dot(betas) is regressing out the effect
        Q, R, _ = linalg.qr(confounds, mode="economic", pivoting=True)
        Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float64).eps * 100.0]
        sn_flat_T -= Q.dot(Q.T).dot(sn_flat_T)
        # print(sn_flat_T.shape)
        # print(sn_flat_T)
        # quit()

        sn_conn[trils[0], trils[1], :] = sn_flat_T.T
        sn_conn[trils[1], trils[0], :] = sn_flat_T.T

    return conn_trials

def analyze_vendor():
    sn_roi_act, sns = pickle_wrap(load_resting_data, easy_override=False)
    bad_rs_sns = {'133'}
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    sn_roi_act = stdize(sn_roi_act, axis=2, nans=True)

    conn_trials = sn_roi_act[..., None, :] * \
                  sn_roi_act[..., None, :, :]

    # conn_trials = high_variance_conn_confounds(conn_trials)
    conn_trials = conn_trials[:, None, :, :, :]
    diag = np.diag_indices(conn_trials.shape[3])
    conn_trials[:, 0, diag[0], diag[1], :] = np.nan

    for i in range(conn_trials.shape[0]):
        sn_conn = conn_trials[i, 0, :, :, :]
        rs = []
        rand = np.random.randint(0, sn_conn.shape[0], (4, 100))
        for x0, y0, x1, y1 in rand.T:
            try:
                r, p = stats.pearsonr(sn_conn[x0, y0, :], sn_conn[x1, y1, :])
            except ValueError:
                continue
            rs.append(r)
        print(f'{np.mean(rs)=:.3f}')
    quit()

    # print(conn_trials.shape)
    # quit()
    # corr = np.mean(conn_trials, axis=(0, 1, 4))
    # atlas = get_atlas()
    # plot_connectivity(corr,
    #                   atlas['ticks'],
    #                   atlas['tick_labels'],
    #                   atlas['tick_lows'],
    #                   no_avg=True,
    #                   title='Resting state connectivity',
    #                   cbar_label='r',
    #                   vmin=-.3, vmax=1.0)
    # quit()

    # print(conn_trials.shape)
    #
    # for i in range(conn_trials.shape[0]):
    #     rs = []
    #     for j in range(2, 200):
    #         r, p = stats.pearsonr(conn_trials[i, 0, 0, 1, :],
    #                               conn_trials[i, 0, 0, j, :])
    #         rs.append(r)
    #     r = np.mean(rs)
    #     print(f'{r=:.3f}')
    # quit()
    #

    print(conn_trials.shape)
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=False, scrub=False)


    conn_keys = ['dd', 'vv', 'dv_ant', 'dv_pos', 'dpva', 'vpda']
    conn_ps = [(p_d_pos, p_d_ant), (p_v_pos, p_v_ant), (p_d_ant, p_v_ant),
               (p_d_pos, p_v_pos), (p_d_pos, p_v_ant), (p_v_pos, p_d_ant)]
    key2conn = {}
    for key, (p0, p1) in zip(conn_keys, conn_ps):
        key2conn[key] = get_module_cross_trialwise_z(conn_trials, p0, p1)

    act_keys = ['dp', 'da', 'vp', 'va']
    act_p = [p_d_pos, p_d_ant, p_v_pos, p_v_ant]
    key2p_M = {}
    for key, p in zip(act_keys, act_p):
        key2p_M[key] = np.nanmean(sn_roi_act[:, p, :], axis=1)

    df_as_d = defaultdict(list)
    n_TRs = sn_roi_act.shape[-1]
    for i, sn in enumerate(sns):
        for key, conn in key2conn.items():
            df_as_d[key].extend(stats.zscore(conn[i, :]))
        for key, M in key2p_M.items():
            df_as_d[key].extend(stats.zscore(M[i, :]))
        df_as_d['sn'].extend([sn] * n_TRs)
    df = pd.DataFrame(df_as_d)
    # keys = act_keys + conn_keys
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    # df['dFC'] = df[conn_keys].sum(axis=1)

    print(df[act_keys + conn_keys].corr())
    quit()

    df['age'] = df['sn'].apply(lambda sn: int(str(sn)[0]))
    df['horz'] = df['dd'] + df['vv']
    df['vert'] = df['dv_ant'] + df['dv_pos']
    df['age'] = stats.zscore(df['age'], nan_policy='omit')
    formula = ('dv_ant ~ vv + dd + dv_pos + dpva + vpda + '
               '(1 | sn)')

    from pymer4 import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())



if __name__ == '__main__':
    analyze_vendor()
    # # load_resting_data()
    quit()

    dir_in = r'Z:\Cabeza\SchemRep.01\Data\fMRIprep_by_subject_out'
    sns = get_sns()
    sns = sns[1] + sns[2]
    for sn in tqdm(sns, desc='Looping over fMRI'):
        has_match = False
        for i in range(1, 4):
            dir_sn = rf'{dir_in}/sub-{sn}/ses-{i}/func'
            d = Path(dir_sn)
            # fp = f'{dir_sn}/sub-{sn}_ses-{i}_task-resting_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
            fp = f'{dir_sn}/sub-{sn}_ses-{i}_task-resting_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

            match = os.path.isfile(fp)
            if match:
                fp_out = fr'E:/PycharmProjects_E/SchemeRep/fMRI_in/{sn}/resting/rs0.nii.gz'
                if os.path.isfile(fp_out):
                    print(f'Already exists: {sn}')
                    break
                Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
                print(f'Copying: {sn}')
                shutil.copyfile(fp, fp_out)
                break
        else:
            print(f'{sn}: no match')