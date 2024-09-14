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


os.chdir(r'C:\PycharmProjects\SchemeRep')

def get_df_events(sn, RL_LR, cont_PE=None):
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


        med_PE = df_trials['PE_abs'].median()
        df_trials['trial_type'] = df_trials['PE_abs'].apply(
            lambda x: 'low_PE' if x < med_PE else 'high_PE')

        pd.set_option('display.max_rows', 2000)
        # print(f'{med_PE=}')
        # print(df_trials[['event', 'trial_type', 'PE_abs']])
        # quit()


    df_trials.drop(columns=['same'], inplace=True)

    return df_trials

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

def do_LSA(img, df_trials, sn, lr):

    df_trials.reset_index(drop=True, inplace=True)

    condition_counter = defaultdict(lambda: 0)
    for i_trial, trial in df_trials.iterrows():
        trial_condition = trial["trial_type"]
        # if 'scn' in trial_condition: continue
        condition_counter[trial_condition] += 1
        trial_name = f"{trial_condition}__{condition_counter[trial_condition]:03d}"
        df_trials.loc[i_trial, "trial_type"] = trial_name
    # print(df_trials)
    # quit()

    frame_times = np.linspace(0, 192, 253, endpoint=False)
    df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2))
    df_motion = load_motion(sn, lr)
    df_confounds = pd.concat([df_compcor, df_motion], axis=1)
    X1 = make_first_level_design_matrix(
        frame_times,
        df_trials,
        add_regs=df_confounds,
        hrf_model='spm',  #
    )

    fp_mask = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\brainmask_fs.2.nii.gz'

    img = image.load_img(img)
    mask = image.load_img(fp_mask)
    global_signal = img.get_fdata()[mask.get_fdata() > 0].mean(axis=0)
    df_confounds['global'] = global_signal
    # print(list(global_signal))
    # quit()

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

    fp_lsa = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_{lr}_LSA.nii'
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


def LSS_gambling(lsa=True, easy_override=False):
    sns = os.listdir(r'G:\HCP_gambling')
    sns = list(sns)
    print(f'{len(sns)=}')
    sns = sorted(sns)
    # sns = sns
    sns = sns[::2]
    sns = sns[::-1]

    bad_sns = []
    for sn in sns:
        try:
            fp_img = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_LR\tfMRI_GAMBLING_LR.nii.gz'
            if lsa:
                fp_lsa_lr = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_lr_LSA.nii'
                if not os.path.exists(fp_lsa_lr) or easy_override:
                    df_events_LR = get_df_events(sn, 'LR')
                    do_LSA(fp_img, df_events_LR, sn, 'LR')
            else:
                fp_lss_lr = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\{sn}_lr_LSS.nii'
                if not os.path.exists(fp_lss_lr) or easy_override:
                    df_events_LR = get_df_events(sn, 'LR')
                    do_LSS(fp_img, df_events_LR, sn, 'LR')

            fp_img = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\tfMRI_GAMBLING_RL.nii.gz'
            if lsa:
                fp_lsa_rl = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_rl_LSA.nii'
                if not os.path.exists(fp_lsa_rl) or easy_override:
                    df_events_RL = get_df_events(sn, 'RL')
                    do_LSA(fp_img, df_events_RL, sn, 'RL')
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


def make_conn(combine_regions=False, bilateral=False, drop_neut=False,
              neut_as_PE=False, regr_M=True, only=None, cont_PE=None):

    fns = os.listdir(r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA')
    sns = {fn.split('_')[0] for fn in fns}
    sns = sorted(list(sns))
    conn_highs = []
    conn_lows = []
    for sn in tqdm(sns, desc='Making conn'):
        try:
            ar = get_sn_roi_ar(sn, 'LR', combine_regions=combine_regions,
                                bilateral=bilateral)
            df_lr = get_df_events(sn, 'LR', cont_PE=cont_PE)
        except ValueError:
            print(f'Not analyzed connectivity: {sn}')
            continue
        except Exception as e:
            print(f'ERROR: {sn}, {e=}')
            time.sleep(1)
            continue


        # df_lr = df_lr[df_lr['event'] != 'neut']'
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
            df_rl = get_df_events(sn, 'RL', cont_PE=cont_PE)
        except ValueError:
            print(f'Not analyzed connectivity: {sn}')
            continue
        except Exception as e:
            print(f'ERROR: {sn}, {e=}')
            time.sleep(10)
            continue
        if only:
            df_rl.loc[df_rl['event'] != only, 'trial_type'] = 'only'
        if drop_neut or neut_as_PE:
            df_rl.loc[df_rl['event'] == 'neut', 'trial_type'] = 'neut'

        # print(df_rl[['event', 'block']].value_counts())
        # print(df_lr[['event', 'block']].value_counts())
        # quit()

        if regr_M:
            ar -= ar.mean(axis=1, keepdims=True)
        if neut_as_PE:
            ar_high2 = ar[:, df_rl['trial_type'] == 'neut']
        else:
            ar_high2 = ar[:, df_rl['trial_type'] == 'high_PE']
        ar_low2 = ar[:, df_rl['trial_type'] == 'low_PE']


        # if drop_neut or neut_as_PE:
        #     # TODO: account for unequal in two sessions for high PE
        #     pass
            # assert ar_high.shape[1] * 6 == ar_low.shape[1]
            # assert ar_high2.shape[1] * 6 == ar_low2.shape[1]
            # ar_low = ar_low[:, ::6]
            # ar_low2 = ar_low2[:, ::6]
        # else:
            # assert ar_high.shape[1] * 3 == ar_low.shape[1]
            # assert ar_high2.shape[1] * 3 == ar_low2.shape[1]
            # ar_low = ar_low[:, ::3]
            # ar_low2 = ar_low2[:, ::3]

        # print(f'{ar_high.shape=}')
        # print(f'{ar_low.shape=}')
        # print(f'{ar_high2.shape=}')
        # print(f'{ar_low2.shape=}')
        # quit()

        ar_high = np.concatenate([ar_high, ar_high2], axis=1)
        ar_low = np.concatenate([ar_low, ar_low2], axis=1)

        conn_high = np.corrcoef(ar_high)
        conn_high[np.diag_indices_from(conn_high)] = np.nan
        conn_low = np.corrcoef(ar_low)
        conn_low[np.diag_indices_from(conn_low)] = np.nan
        # plt.imshow(conn_high)
        # plt.colorbar()
        # plt.show()
        # quit()
        # print(f'{ar_high.shape=}')
        # print(f'{ar_low.shape=}')
        # quit()

        # TODO: Lateralized connectivity.
        #  high R-A/high R-P and low L-A/low L-P means A-P connectivity
        conn_highs.append(conn_high)
        conn_lows.append(conn_low)
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
    if combine_bilateral:
        p_d_ant = np.array(p_d_ant[::2]) // 2
        p_d_pos = np.array(p_d_pos[::2]) // 2
        p_v_ant = np.array(p_v_ant[::2]) // 2
        p_v_pos = np.array(p_v_pos[::2]) // 2

    dd = conn[:, *np.ix_(p_d_pos, p_d_ant)]
    dd = np.nanmean(dd, axis=(1, 2))
    vv = conn[:, *np.ix_(p_v_pos, p_v_ant)]
    vv = np.nanmean(vv, axis=(1, 2))
    dv_ant = conn[:, *np.ix_(p_d_ant, p_v_ant)]
    dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    dv_pos = conn[:, *np.ix_(p_d_pos, p_v_pos)]
    dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    # return dv_ant
    return dd + vv - dv_ant - dv_pos


def test_vendor(combine_regions=False, bilateral=False, corr_z=True):
    conn_highs, conn_lows, sns = (
        pickle_wrap(make_conn, kwargs={'combine_regions': combine_regions,
                                       'bilateral': bilateral,
                                       'neut_as_PE': None,
                                       'drop_neut': False,
                                       'regr_M': False,
                                       'only': 'win',
                                       'cont_PE': 0.5},
                                       easy_override=False))
    dif = conn_highs - conn_lows
    M = np.nanmean(dif, axis=0)
    SE = stats.sem(dif, axis=0, nan_policy='omit')
    t = M / SE

    # t[np.abs(t) > 3] = np.nan

    # print(t.shape)
    if corr_z:
        t_flat = t[np.tril_indices_from(t, k=-1)]
        z_both = get_SchemeRep_regr(combine_regions=combine_regions)
        z_flat = z_both[np.tril_indices_from(z_both, k=-1)]
        r, p = stats.spearmanr(t_flat, z_flat, nan_policy='omit')
        print(f'Gambling x SchemeRep: {r=:.2f}, {p=:.3f}')
        # quit()

    if not bilateral:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=bilateral, HCP=True,
                          lifu_labels=False)
        # t = np.nanmean(dif > 0, axis=0)
        # t[t > 0] = 1
        # t[t < 0] = -1
        plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title='Gambling', tile=.01,
                          no_avg=True, cbar_label='Correlation (r)')


    ef_high = get_vd_ef(conn_highs, combine_regions=combine_regions,
                        combine_bilateral=bilateral)
    ef_low = get_vd_ef(conn_lows, combine_regions=combine_regions,
                       combine_bilateral=bilateral)
    itr = ef_low - ef_high
    t, p = stats.ttest_1samp(itr, 0)
    N = itr.shape[0]
    print(f't[{N - 1}] = {t:.2f}, {p=:.3f}')
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


def get_SchemeRep_regr(regress=False, combine_regions=False):
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
    plot_connectivity(z_both, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'], title='SchemeRep matrix', tile=.01,
                      no_avg=True, cbar_label='Correlation (r)')
    # if combine_regions:
    #     # print(z_both.shape)
    #     # print(len(atlas['ROIs']))
    #     z_both = z_both[np.ix_(len(atlas['ROIs']), len(atlas['ROIs']))]
    # print(z_both.shape)
    # quit()


    return z_both


if __name__ == '__main__':
    # get_SchemeRep_regr()
    # fp = r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\100206_LR_LSS.nii'
    # img = image.load_img(fp)
    # print(img.shape)
    # LSS_gambling()
    test_vendor()
    # make_conn()
    # unzip_all_gambling()
    # test_LSS_x_LSA()






