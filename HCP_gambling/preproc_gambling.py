import zipfile
import os
from collections import defaultdict
from copy import deepcopy

import pandas as pd
from nilearn.image import high_variance_confounds
from tqdm import tqdm

from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
import numpy as np
from nilearn import image
import scipy.stats as stats
from nilearn import plotting

import matplotlib.pyplot as plt

from atlas_utils import get_atlas
from networks.old.network_funcs import load_FC_for_Lifu
from networks.vendor_partitioning import get_vendor_partitions, do_regression
# from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from utils import pickle_wrap
# from vendor_partitioning import get_vendor_partitions, do_regression
import time
from numba import jit, prange, njit

os.chdir(r'C:\PycharmProjects\SchemeRep')

def get_df_events(sn, RL_LR, cont_PE=None, cont_pe_by_event=False):
    dir_LR = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{RL_LR}\EVs'
    df_loss_event = pd.read_csv(fr'{dir_LR}\loss_event.txt', delimiter='\t',
                                header=None, names=['onset', 'duration', 'amplitude'])
    df_loss_blocks = pd.read_csv(fr'{dir_LR}\loss.txt', delimiter='\t',
                                 header=None, names=['onset', 'duration', 'amplitude'])
    df_win_event = pd.read_csv(fr'{dir_LR}\win_event.txt', delimiter='\t',
                               header=None, names=['onset', 'duration', 'amplitude'])
    df_win_blocks = pd.read_csv(fr'{dir_LR}\win.txt', delimiter='\t',
                                header=None, names=['onset', 'duration', 'amplitude'])
    df_neut_events = pd.read_csv(fr'{dir_LR}\neut_event.txt', delimiter='\t',
                                 header=None, names=['onset', 'duration', 'amplitude'])

    t2block = {t: 'loss' for t in df_loss_blocks['onset']}
    t2block.update({t: 'win' for t in df_win_blocks['onset']})

    def get_block(t):
        for t_block, block in t2block.items():
            if t_block <= t < t_block + 40:
                return block
        else:
            raise ValueError

    df_loss_event['event'] = 'loss'
    df_win_event['event'] = 'win'
    df_neut_events['event'] = 'neut'

    df_trials = pd.concat([df_loss_event, df_win_event, df_neut_events])
    df_trials.sort_values('onset', inplace=True)
    df_trials['block'] = df_trials['onset'].apply(get_block)
    df_trials['same'] = df_trials['event'] == df_trials['block']
    df_trials['trial_type'] = df_trials['same'].apply(
        lambda x: 'low_PE' if x else 'high_PE')
    df_trials.reset_index(drop=True, inplace=True)
    df_trials['block_num'] = df_trials.index // 8

    if cont_PE is not None:
        PE_cont = []
        E = 0
        learning = cont_PE
        for idx, row in df_trials.iterrows():
            if row['event'] == 'win':
                PE = 1 - E
                E = E * (1 - learning) + learning
            elif row['event'] == 'loss':
                PE = -1 - E
                E = E * (1 - learning) - learning
            else:
                PE = 0 - E
                E = E * (1 - learning)
            PE_cont.append(PE)
        #     print(row['event'], f'{PE=:.2f}, {E=:.2f}')
        # quit()
        # df_trials['E'] = E
        df_trials['PE'] = PE_cont
        df_trials['PE_abs'] = df_trials['PE'].abs()

        if cont_pe_by_event:
            for event in ['loss', 'win', 'neut']:
                df_event = df_trials[df_trials['event'] == event]
                med_PE = df_event['PE_abs'].mean()
                # print(f'{event}: {med_PE=}')
                df_trials.loc[df_trials['event'] == event, 'trial_type'] = df_trials.loc[
                    df_trials['event'] == event, 'PE_abs'].apply(
                    lambda x: 'low_PE' if x < med_PE else 'high_PE')
        else:
            med_PE = df_trials['PE_abs'].median()
            df_trials['trial_type'] = df_trials['PE_abs'].apply(
                lambda x: 'low_PE' if x < med_PE else 'high_PE')

        pd.set_option('display.max_rows', 2000)

    df_trials.drop(columns=['same'], inplace=True)

    return df_trials

# df_trials = get_df_events('139435', 'rl')
# print(df_trials)
# df_trials = get_df_events('139435', 'lr')
# print(df_trials)
#
# quit()


def lss_transformer(df, row_number):
    """Label one trial for one LSS model.

    Parameters
    ----------
    df : pandas.DataFrame
        BIDS-compliant events file information.
    row_number : int
        Row number in the DataFrame.
        This indexes the trial that will be isolated.

    Returns
    -------
    df : pandas.DataFrame
        Update events information, with the select trial's trial type isolated.
    trial_name : str
        Name of the isolated trial's trial type.
    """
    df = df.copy()

    # Determine which number trial it is *within the condition*
    trial_condition = df.loc[row_number, "trial_type"]
    trial_type_series = df["trial_type"]
    trial_type_series = trial_type_series.loc[
        trial_type_series == trial_condition
    ]
    trial_type_list = trial_type_series.index.tolist()
    trial_number = trial_type_list.index(row_number)

    # We use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names.
    # Technically, all you need is for the requested trial to have a unique
    # 'trial_type' *within* the dataframe, rather than across models.
    # However, we may want to have meaningful 'trial_type's (e.g., 'Left_001')
    # across models, so that you could track individual trials across models.
    trial_name = f"{trial_condition}__{trial_number:03d}"
    df.loc[row_number, "trial_type"] = trial_name
    return df, trial_name

def load_motion(sn, lr):
    fp_motion = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\Movement_Regressors.txt'
    motion = np.loadtxt(fp_motion)[:, :12]
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz",
                     "dtx", "dty", "dtz", "drx", "dry", "drz"]
    df = pd.DataFrame(motion, columns=add_reg_names)
    return df

def do_LSA(img, df_trials, sn, lr, reg_global=False):

    df_trials.reset_index(drop=True, inplace=True)

    condition_counter = defaultdict(lambda: 0)
    for i_trial, trial in df_trials.iterrows():
        trial_condition = trial["trial_type"]
        # if 'scn' in trial_condition: continue
        condition_counter[trial_condition] += 1
        trial_name = f"{trial_condition}__{condition_counter[trial_condition]:03d}"
        df_trials.loc[i_trial, "trial_type"] = trial_name

    frame_times = np.linspace(0, 192, 253, endpoint=False)
    df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2))
    df_motion = load_motion(sn, lr)
    df_confounds = pd.concat([df_compcor, df_motion], axis=1)

    if reg_global:
        fp_mask = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\brainmask_fs.2.nii.gz'
        img = image.load_img(img)
        mask = image.load_img(fp_mask)
        global_signal = img.get_fdata()[mask.get_fdata() > 0].mean(axis=0)
        df_confounds['global'] = global_signal
        # print(global_signal)


    X1 = make_first_level_design_matrix(
        frame_times,
        df_trials,
        add_regs=df_confounds,
        hrf_model='spm',  #
    )

    # MBs = process.memory_info().rss / 1024 / 1024
    # logging.debug(f'Making first level model: {MBs=:.2f}')
    glm = FirstLevelModel(slice_time_ref=0.5, t_r=192 / 253,
                          signal_scaling=(0, 1), high_pass=1 / 128,
                          minimize_memory=True,
                          mask_img=fp_mask,
                          verbose=100)


    # MBs = process.memory_info().rss / 1024 / 1024
    # logging.debug(f'Fitting first level model: {MBs=:.2f}')
    print('Fitting LSA model...')
    glm = glm.fit(img, design_matrices=X1)

    # plotting.plot_design_matrix(glm.design_matrices_[0])
    # plt.show()
    # quit()
    # MBs = process.memory_info().rss / 1024 / 1024
    # logging.debug(f'Succesfully fit: {MBs=:.2f}')
    del X1
    betas = []
    print('Making contrasts...')
    for i_trial, trial in df_trials.iterrows():
        trial_name = trial["trial_type"]
        # print(f'{trial=}, {trial_name=}')
        # if 'obj__' not in trial_name:
        #     continue
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Onto trial {i_trial}, {trial}: {MBs=:.2f}')

        # print(f'Commute contrast: {i_trial} | {trial_name}')
        beta_map = glm.compute_contrast(trial_name,
                                        output_type='effect_size')

        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Getting beta map fdata {beta_map.shape=}: {MBs=:.2f}')

        beta = beta_map.get_fdata()[..., 0]
        del beta_map
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Making a deep copy: {MBs=:.2f}')
        betas.append(deepcopy(beta))
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Appended beta deepcopy: {MBs=:.2f}')

    betas = np.array(betas)
    betas = np.transpose(betas, (1, 2, 3, 0))

    beta_img = image.new_img_like(img, betas)

    glob_str = '_global' if reg_global else ''
    fp_lsa = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_{lr}_LSA{glob_str}.nii'
    beta_img.to_filename(fp_lsa)
    # quit()


def do_LSS(img, df_trials, sn, lr):
    fp_mask = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\brainmask_fs.2.nii.gz'
    df_trials.reset_index(drop=True, inplace=True)
    betas = []
    df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2))
    df_motion = load_motion(sn, lr)
    df_confounds = pd.concat([df_compcor, df_motion], axis=1)
    for i in tqdm(range(len(df_trials)), desc=f'Cooking LSS: {sn} ({lr})',
                  position=0, leave=True):

        df_trial, i_name = lss_transformer(df_trials, i)

        frame_times = np.linspace(0, 192, 253, endpoint=False)

        X1 = make_first_level_design_matrix(
            frame_times,
            df_trial,
            # drift_model="polynomial",
            # drift_order=3,
            add_regs=df_confounds,
            hrf_model='spm',  #  + derivative + dispersion
        )
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Making first level model: {MBs=:.2f}')
        glm = FirstLevelModel(slice_time_ref=0.5, t_r=192/253,
                              signal_scaling=(0, 1), high_pass=1 / 128,
                              minimize_memory=True,
                              mask_img=fp_mask,
                              verbose=100)
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Fitting first level model: {MBs=:.2f}')
        glm = glm.fit(img, design_matrices=X1)
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Computing contrast: {MBs=:.2f}')
        beta_map = glm.compute_contrast(i_name,
                                        output_type='effect_size')
        del glm
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Getting beta map fdata {beta_map.shape=}: {MBs=:.2f}')
        beta = beta_map.get_fdata()[..., 0]
        del beta_map
        del X1
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Making a deep copy: {MBs=:.2f}')
        betas.append(deepcopy(beta))
        # MBs = process.memory_info().rss / 1024 / 1024
        # logging.debug(f'Appended beta deepcopy: {MBs=:.2f}')

    betas = np.array(betas)
    betas = np.transpose(betas, (1, 2, 3, 0))

    beta_img = image.new_img_like(img, betas)

    fp_lss = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\{sn}_{lr}_LSS.nii'
    beta_img.to_filename(fp_lss)


def LSS_gambling(lsa=True, easy_override=False, reg_global=True):
    sns = os.listdir(r'G:\HCP_gambling')
    sns = list(sns)
    print(f'{len(sns)=}')
    sns = sorted(sns)
    # sns = sns
    # sns = sns[1::2]
    # sns = sns[::-1]
    # sns = sns[len(sns)//2:]
    # sns = ['100206']
    # easy_override = True

    glob_str = '_global' if reg_global else ''


    bad_sns = []
    for sn in sns:
        try:
            fp_img = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_LR\tfMRI_GAMBLING_LR.nii.gz'
            if lsa:
                fp_lsa_lr = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_lr_LSA{glob_str}.nii'
                if not os.path.exists(fp_lsa_lr) or easy_override:
                    df_events_LR = get_df_events(sn, 'LR')
                    do_LSA(fp_img, df_events_LR, sn, 'LR', reg_global=reg_global)
            else:
                fp_lss_lr = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\{sn}_lr_LSS.nii'
                if not os.path.exists(fp_lss_lr) or easy_override:
                    df_events_LR = get_df_events(sn, 'LR')
                    do_LSS(fp_img, df_events_LR, sn, 'LR')

            fp_img = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\tfMRI_GAMBLING_RL.nii.gz'
            if lsa:
                fp_lsa_rl = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_rl_LSA{glob_str}.nii'
                if not os.path.exists(fp_lsa_rl) or easy_override:
                    df_events_RL = get_df_events(sn, 'RL')
                    do_LSA(fp_img, df_events_RL, sn, 'RL', reg_global=reg_global)
            else:
                fp_lss_rl = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\{sn}_rl_LSS.nii'
                if not os.path.exists(fp_lss_rl) or easy_override:
                    df_events_RL = get_df_events(sn, 'RL')
                    do_LSS(fp_img, df_events_RL, sn, 'RL')
            print(f'Done: {sn}')
        except FileNotFoundError:
            print(f'File not found: {sn}')
            time.sleep(1)
            bad_sns.append(sn)
        except ValueError as e:
            print(f'ValueError: {sn}, {e=}')
            bad_sns.append(sn)
        except Exception as e:
            print(f'ERROR: {sn=}, {e=}')
            time.sleep(1)
            bad_sns.append(sn)
    print(f'{bad_sns=}')

def get_sn_roi_ar(sn, lr, combine_regions=False, bilateral=False):
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=bilateral,
                      HCP=True)

    fp_lsa_lr = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_{lr}_LSA.nii'
    img_lsa_lr = image.load_img(fp_lsa_lr)
    data_lsa_lr = img_lsa_lr.get_fdata()
    df_events = get_df_events(sn, 'LR')

    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    ROI2vecs = {}
    region2vecs = defaultdict(list)
    ar = []
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        roi_data_lsa_lr = data_lsa_lr[atlas_roi]
        vals = roi_data_lsa_lr.mean(axis=0)
        # print(vals.shape)
        # quit()

        ar.append(vals)
    ar = np.array(ar)
    return ar

def get_conn_sn(sn, combine_regions=False, bilateral=False, drop_neut=False,
              neut_as_PE=False, regr_M=True, only=None, cont_PE=None,
              cont_PE_by_event=False, lr_separate=False):

    try:
        ar = get_sn_roi_ar(sn, 'LR', combine_regions=combine_regions,
                           bilateral=bilateral)
        df_lr = get_df_events(sn, 'LR', cont_PE=cont_PE,
                              cont_pe_by_event=cont_PE_by_event)
    except ValueError:
        print(f'Not analyzed connectivity: {sn}')
        return None, sn
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        # bad_sns.append(sn)
        time.sleep(1)
        return None, sn
    # print(df_lr)
    # quit()

    if only:
        df_lr.loc[df_lr['event'] != only, 'trial_type'] = 'only'
    if drop_neut or neut_as_PE:
        df_lr.loc[df_lr['event'] == 'neut', 'trial_type'] = 'neut'

    if regr_M:
        ar -= ar.mean(axis=1, keepdims=True)
    if neut_as_PE:
        ar_high = ar[:, df_lr['trial_type'] == 'neut']
    else:
        ar_high = ar[:, df_lr['trial_type'] == 'high_PE']

    ar_low = ar[:, df_lr['trial_type'] == 'low_PE']

    try:
        ar = get_sn_roi_ar(sn, 'RL', combine_regions=combine_regions,
                           bilateral=bilateral)
        df_rl = get_df_events(sn, 'RL', cont_PE=cont_PE,
                              cont_pe_by_event=cont_PE_by_event)
    except ValueError:
        print(f'Not analyzed connectivity: {sn}')
        return None, sn
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        time.sleep(1)
        # bad_sns.append(sn)
        return None, sn
    if only:
        df_rl.loc[df_rl['event'] != only, 'trial_type'] = 'only'
    if drop_neut or neut_as_PE:
        df_rl.loc[df_rl['event'] == 'neut', 'trial_type'] = 'neut'

    if regr_M:
        ar -= ar.mean(axis=1, keepdims=True)
    if neut_as_PE:
        ar_high2 = ar[:, df_rl['trial_type'] == 'neut']
    else:
        ar_high2 = ar[:, df_rl['trial_type'] == 'high_PE']
    ar_low2 = ar[:, df_rl['trial_type'] == 'low_PE']
    # print(df_rl)
    # quit()
    if lr_separate:
        conn_high0 = np.corrcoef(ar_high)
        conn_high0[np.diag_indices_from(conn_high0)] = np.nan
        conn_low0 = np.corrcoef(ar_low)
        conn_low0[np.diag_indices_from(conn_low0)] = np.nan
        conn_high1 = np.corrcoef(ar_high2)
        conn_high1[np.diag_indices_from(conn_high1)] = np.nan
        conn_low1 = np.corrcoef(ar_low2)
        conn_low1[np.diag_indices_from(conn_low1)] = np.nan
        conn_high = (conn_high0 + conn_high1) / 2
        conn_low = (conn_low0 + conn_low1) / 2
    else:
        ar_high = np.concatenate([ar_high, ar_high2], axis=1)
        ar_low = np.concatenate([ar_low, ar_low2], axis=1)
        print(f'{ar_high.shape=} | {ar_low.shape=}')

        conn_high = np.corrcoef(ar_high)
        conn_high[np.diag_indices_from(conn_high)] = np.nan
        conn_low = np.corrcoef(ar_low)
        conn_low[np.diag_indices_from(conn_low)] = np.nan
    # print(conn_low1)
    # quit()

    # TODO: Lateralized connectivity.
    #  high R-A/high R-P and low L-A/low L-P means A-P connectivity
    return conn_high, conn_low

def pwrap_get_conn_sn(sn, **kw):
    conn_high, conn_low_sn = pickle_wrap(get_conn_sn, kwargs=kw,
                                         easy_override=False,
                                         )

def make_conn(combine_regions=False, bilateral=False, drop_neut=False,
              neut_as_PE=False, regr_M=True, only=None, cont_PE=None,
              cont_PE_by_event=False, lr_separate=True, num_sns=None,
              n_jobs=1):

    fns = os.listdir(r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA')
    sns = {fn.split('_')[0] for fn in fns}
    sns = sorted(list(sns))
    if sns is not None:
        sns = sns[:num_sns]
    # sns = sns[:-1]
    # sns = sns[::-2]

    conn_highs = []
    conn_lows = []
    bad_sns = []
    kw = {'combine_regions': combine_regions,  'bilateral': bilateral,
          'neut_as_PE': neut_as_PE, 'drop_neut': drop_neut, 'regr_M': regr_M,
          'only': only, 'cont_PE': cont_PE, 'cont_PE_by_event': cont_PE_by_event,
          'lr_separate': lr_separate}

    if cont_PE_by_event:
        from datetime import datetime
        dt_max = datetime(2024, 9, 15, 11, 0, 0)
    else:
        dt_max = None

    if n_jobs > 1:
        from multiprocessing import Pool
        from functools import partial
        get_conn_sn_partial = partial(get_conn_sn, **kw)

        with Pool(n_jobs) as p:
            res = tqdm(p.imap(get_conn_sn_partial, sns), desc='Parallel get_conn_sn')
        for conn_high, conn_low_sn in res:
            if conn_high is None:
                bad_sns.append(conn_low_sn)
                continue
            conn_highs.append(conn_high)
            conn_lows.append(conn_low_sn)
        # TODO: maybe finish this

    for sn in tqdm(sns, desc='Making conn'):
        kw['sn'] = sn
        conn_high, conn_low_sn = pickle_wrap(get_conn_sn, kwargs=kw,
                                             easy_override=False,
                                             dt_max=dt_max)
        if conn_high is None:
            bad_sns.append(sn)
            continue

        conn_highs.append(conn_high)
        conn_lows.append(conn_low_sn)

    print(f'{bad_sns=}')
    conn_highs = np.array(conn_highs)
    conn_lows = np.array(conn_lows)

    return conn_highs, conn_lows, sns

def test_LSS_x_LSA():
    fp_LSS = r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\100206_LR_LSS.nii'
    img_LSS = image.load_img(fp_LSS).get_fdata()
    fp_LSA = r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\100206_LR_LSA.nii'
    img_LSA = image.load_img(fp_LSA).get_fdata()

    for _ in range(100):
        rand_x = np.random.randint(0, img_LSS.shape[0])
        rand_y = np.random.randint(0, img_LSS.shape[1])
        rand_z = np.random.randint(0, img_LSS.shape[2])
        if np.any(np.isnan(img_LSS[rand_x, rand_y, rand_z, :])):
            continue
        if np.any(img_LSS[rand_x, rand_y, rand_z, :] == 0):
            continue
        r, p = stats.spearmanr(img_LSS[rand_x, rand_y, rand_z, :],
                               img_LSA[rand_x, rand_y, rand_z, :])
        print(f'{r=:.2f}')
        # plt.scatter(img_LSS[rand_x, rand_y, rand_z, :],
        #             img_LSA[rand_x, rand_y, rand_z, :])
        # plt.show()
        # quit()
        # print(f'{img_LSS[rand_x, rand_y, rand_z, :]=} | {img_LSA[rand_x, rand_y, rand_z, :]=}')
            # print(f'{img_LSS[rand_x, rand_y, rand_z]=} | {img_LSA[rand_x, rand_y, rand_z]=}')
            # print(f'{img_LSS[rand_x, rand_y, rand_z] - img_LSA[rand_x, rand_y, rand_z]}')
            # break
    quit()

def get_vd_ef(conn, combine_regions=False, combine_bilateral=False):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', anat=True, weighted=False,
                              flip=True, thr=.9, scrub=False, anat_ver=3,
                              combine_regions=combine_regions)
    print(f'{p_d_ant=},\n{p_d_pos=},\n{p_v_ant=},\n{p_v_pos=}')
    # print(p_d_ant)
    # print(p_v_pos)
    # print(p_v_ant)
    # quit()



    if combine_bilateral:
        p_d_ant = np.array(p_d_ant[::2]) // 2
        p_d_pos = np.array(p_d_pos[::2]) // 2
        p_v_ant = np.array(p_v_ant[::2]) // 2
        p_v_pos = np.array(p_v_pos[::2]) // 2

    dd = conn[:, *np.ix_(p_d_pos, p_d_ant)]
    # print(dd[89])
    # quit()
    dd = np.nanmean(dd, axis=(1, 2))
    vv = conn[:, *np.ix_(p_v_pos, p_v_ant)]
    vv = np.nanmean(vv, axis=(1, 2))
    dv_ant = conn[:, *np.ix_(p_d_ant, p_v_ant)]
    dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    dv_pos = conn[:, *np.ix_(p_d_pos, p_v_pos)]
    dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    M_overall = np.nanmean(conn, axis=(1, 2))


    # return dv_ant
    return dd - dv_ant
    # return vv - dv_pos
    # return dv_pos

    return dd + vv - dv_ant - dv_pos# - M_overall

def get_combo(kw):
    kw['only'] = 'loss'
    conn_highs, conn_lows, sns = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    kw['only'] = 'win'
    conn_highs2, conn_lows2, sns2 = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    conn_highs = np.mean([conn_highs, conn_highs2], axis=0)
    conn_lows = np.mean([conn_lows, conn_lows2], axis=0)
    return conn_highs, conn_lows, sns


def test_vendor(combine_regions=False, bilateral=False, corr_z=True,
                sub_ROI_expected=False):
    # OKAY. Keep REGR_R as True. However, it is mostly inconsequential
    # Keep dropping Neut = TRUE
    # keep cont_PE_by_event
    # lr_separate has no effect
    # keep cont_PE = .3
    # cont_PE = 1. sucks. no t effect. some matrix correlation

    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'neut_as_PE': None, 'drop_neut': True, 'only': None,
          'num_sns': 1000, 'cont_PE': 0.30, 'cont_PE_by_event': True,
          'regr_M': True, 'lr_separate': False}


    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'neut_as_PE': None, 'drop_neut': True, 'only': None,
          'num_sns': 1000, 'cont_PE': .3, 'cont_PE_by_event': True,
          'regr_M': True, 'lr_separate': False}


    if kw['neut_as_PE']:
        kw['drop_neut'] = False
        kw['only'] = None
        kw['cont_PE'] = None
        kw['cont_PE_by_event'] = False

    if kw['only'] == 'combo':
        conn_highs, conn_lows, sns = get_combo(kw)
    else:
        conn_highs, conn_lows, sns = (
            pickle_wrap(make_conn, kwargs=kw, easy_override=False))

    if combine_regions:
        conn_highs[:, :, 46:] = np.nan
        conn_highs[:, 46:, :] = np.nan
        conn_lows[:, :, 46:] = np.nan
        conn_lows[:, 46:, :] = np.nan
    else:
        conn_highs[:, :, 210:] = np.nan
        conn_highs[:, 210:, :] = np.nan
        conn_lows[:, :, 210:] = np.nan
        conn_lows[:, 210:, :] = np.nan

    if sub_ROI_expected:
        ROI_expected = np.nanmean(conn_highs, axis=(0, 2))
        ROI_expected = (ROI_expected[:, None] + ROI_expected[None, :]) / 2
        conn_highs -= ROI_expected[None]
        # ROI_expected = np.sqrt(ROI_expected[:, None] * ROI_expected[None, :])
        # conn_highs /= ROI_expected[None]
        ROI_expected = np.nanmean(conn_lows, axis=(0, 2))
        ROI_expected = (ROI_expected[:, None] + ROI_expected[None, :]) / 2
        conn_lows -= ROI_expected[None]
        # ROI_expected = np.sqrt(ROI_expected[:, None] * ROI_expected[None, :])
        # conn_lows /= ROI_expected[None]

    # conn_highs -= np.nanmean(conn_highs, axis=(1, 2), keepdims=True)
    # conn_lows -= np.nanmean(conn_lows, axis=(1, 2), keepdims=True)

    dif = conn_highs - conn_lows
    M = np.nanmean(dif, axis=0)
    SE = stats.sem(dif, axis=0, nan_policy='omit')
    t = M / SE

    if corr_z:
        t_flat = t[np.tril_indices_from(t, k=-1)]
        z_both = get_SchemeRep_regr(combine_regions=combine_regions, plot=False)
        z_flat = z_both[np.tril_indices_from(z_both, k=-1)]
        r, p = stats.spearmanr(t_flat, z_flat, nan_policy='omit')
        print(f'Gambling x SchemeRep: {r=:.2f}, {p=:.3f}')


    if not bilateral:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=bilateral, HCP=True,
                          lifu_labels=False)


        title = str(kw)
        title_ = ''
        for i in range(len(title) // 50):
            title_ += title[i * 50:(i + 1) * 50] + '\n'
        title = title_
        # quit()

        # p_v_pos = [188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
        # p_d_pos = [134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145]
        # plt.imshow(t[np.ix_(p_v_pos, p_d_pos)])
        # plt.colorbar()
        # plt.show()

        plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title=title, tile=.01,
                          no_avg=True, cbar_label='Correlation (r)',
                          vmin=-4, vmax=4)
        # quit()

    ef_high = get_vd_ef(conn_highs, combine_regions=combine_regions,
                        combine_bilateral=bilateral)
    ef_low = get_vd_ef(conn_lows, combine_regions=combine_regions,
                       combine_bilateral=bilateral)
    itr = ef_low - ef_high
    t, p = stats.ttest_1samp(itr, 0)
    N = itr.shape[0]
    nans = np.sum(np.isnan(itr))
    print(f't[{N - nans - 1}/{N - 1}] = {t:.2f}, {p=:.3f}')
    print(kw)
    quit()
    n, bins, patches = plt.hist(itr, range=(-0.4, 0.4), bins=40)
    plt.plot([0, 0], [0, np.max(n)], 'r--')
    plt.show()

    plt.hist(itr, range=(-0.4, 0.4), bins=40,
             cumulative=True, density=True)
    plt.plot([0, 0], [0, 1], 'r--')
    plt.plot([-.4, .4], [0.5, 0.5], 'r--')
    plt.xlim(-0.4, 0.4)
    plt.ylim(0, 1)
    plt.show()
    quit()


def get_SchemeRep_regr(regress=False, combine_regions=False, plot=False):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    z_both = do_regression(sn_inc_conn, flip=False) # False = (Incongruent > Congruent)

    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False, HCP=True,
                      lifu_labels=False)
    if combine_regions:
        z_both = z_both[:54, :54]
    if plot:
        plot_connectivity(z_both, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title='SchemeRep matrix', tile=.01,
                          no_avg=True, cbar_label='Correlation (r)')

    return z_both

@njit(fastmath=True, nopython=True, cache=True, parallel=True)
def numba_corrcoef(X):
    nrow = X.shape[0]
    ncol = X.shape[1]
    out = np.ones((nrow, nrow))
    # Es = np.empty(nrow)
    # E_sqs = np.empty(nrow)
    denom_parts = np.empty(nrow)

    X_mod = np.empty((nrow, ncol))
    for j in range(nrow):
        s = 0
        ss = 0
        for i in range(ncol):
            s += X[j, i]
            ss += X[j, i] ** 2
        M = s / ncol
        SD = np.sqrt(ss / ncol - M ** 2)
        for i in range(ncol):
            X_mod[j, i] = (X[j, i] - M) / SD
        # X_mod = (X[j, :] - M) / SD
    # X = X_mod

    for j in range(nrow):
        s = 0
        ss = 0
        for i in range(ncol):
            s += X_mod[j, i]
            ss += X_mod[j, i] ** 2
        # Es[j] = s / ncol
        # E_sqs[j] = ss / ncol
        denom_parts[j] = np.sqrt(ss / ncol)# - Es[j] ** 2)

    for j in prange(nrow):
        for k in range(j + 1, nrow):
            if j == k:
                out[j, k] = 1
                continue
            prod_sum = 0
            # E_J = Es[j]
            # E_JJ = E_sqs[j]
            # E_K = Es[k]
            # E_KK = E_sqs[k]
            num_points = ncol
            for i in range(ncol):
                prod_sum += (X_mod[j, i] * X_mod[k, i])
            # E_JK = prod_sum / num_points
            # numerator = E_JK# - (Es[j] * Es[k])
            # denominator = denom_parts[j] * denom_parts[k]
            out[j, k] = out[k, j] = prod_sum / num_points# numerator #/ denominator
    return out

def test_corr():
    X = np.random.normal(size=(10000, 50))
    t_st = time.time()
    for _ in tqdm(range(10)):
        corr_np = np.corrcoef(X)
    t_end = time.time()
    t_np = t_end - t_st
    print(f'Numpy: {t_end - t_st:.4f}')

    t_st = time.time()
    corr = numba_corrcoef(X)
    all_same = np.allclose(corr_np, corr)
    print(f'{all_same=}')
    # assert all_same
    t_end = time.time()
    print(f'Numba first: {t_end - t_st:.4f}')

    t_st = time.time()
    for _ in range(10):
        corr = numba_corrcoef(X)
    t_end = time.time()
    t_nb = t_end - t_st
    print(f'Numba: {t_end - t_st:.4f}')
    print(f'Ratio: {t_np/t_nb:.2f}')


if __name__ == '__main__':
    # test_corr()
    # fp = r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\100206_LR_LSS.nii'
    # img = image.load_img(fp)
    # print(img.shape)
    LSS_gambling()
    # test_vendor()
    # make_conn()
    # unzip_all_gambling()
    # test_LSS_x_LSA()






