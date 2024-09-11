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

os.chdir(r'H:\PycharmProjects_H\SchemeRep')

def get_df_events(sn, RL_LR):
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

    df_trials.drop(columns=['event', 'block', 'same'], inplace=True)

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

    fp_lsa = fr'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSA\{sn}_{lr}_LSA.nii'
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

    fp_lss = fr'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSS\{sn}_{lr}_LSS.nii'
    beta_img.to_filename(fp_lss)


def LSS_gambling(lsa=True):
    sns = os.listdir(r'G:\HCP_gambling')
    sns = list(sns)

    sns = sorted(sns)
    print(f'Number of sns: {len(sns)}')
    quit()
    # print(sns)
    # quit()
    for sn in sns:

        fp_img = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_LR\tfMRI_GAMBLING_LR.nii.gz'
        if lsa:
            fp_lsa_lr = fr'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSA\{sn}_lr_LSA.nii'
            if not os.path.exists(fp_lsa_lr):
                df_events_LR = get_df_events(sn, 'LR')
                do_LSA(fp_img, df_events_LR, sn, 'LR')
        else:
            fp_lss_lr = fr'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSS\{sn}_lr_LSS.nii'
            if not os.path.exists(fp_lss_lr):
                df_events_LR = get_df_events(sn, 'LR')
                do_LSS(fp_img, df_events_LR, sn, 'LR')

        fp_img = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\tfMRI_GAMBLING_RL.nii.gz'
        if lsa:
            fp_lsa_rl = fr'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSA\{sn}_rl_LSA.nii'
            if not os.path.exists(fp_lsa_rl):
                df_events_RL = get_df_events(sn, 'RL')
                do_LSA(fp_img, df_events_RL, sn, 'RL')
        else:
            fp_lss_rl = fr'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSS\{sn}_rl_LSS.nii'
            if not os.path.exists(fp_lss_rl):
                df_events_RL = get_df_events(sn, 'RL')
                do_LSS(fp_img, df_events_RL, sn, 'RL')

def get_sn_roi_ar(sn, lr, combine_regions=False, bilateral=False):
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=bilateral,
                      HCP=True)

    fp_lsa_lr = fr'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSA\{sn}_{lr}_LSA.nii'
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


def make_conn(combine_regions=False, bilateral=False):

    fns = os.listdir(r'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSA')
    sns = {fn.split('_')[0] for fn in fns}
    sns = sorted(list(sns))
    for sn in tqdm(sns, desc='Making conn'):
        ar = get_sn_roi_ar(sn, 'LR', combine_regions=combine_regions,
                            bilateral=bilateral)
        df_lr = get_df_events(sn, 'LR')
        ar_high = ar[:, df_lr['trial_type'] == 'high_PE']
        ar_low = ar[:, df_lr['trial_type'] == 'low_PE']

        ar = get_sn_roi_ar(sn, 'RL', combine_regions=combine_regions,
                            bilateral=bilateral)
        df_rl = get_df_events(sn, 'RL')
        ar_high2 = ar[:, df_rl['trial_type'] == 'high_PE']
        ar_low2 = ar[:, df_rl['trial_type'] == 'low_PE']

        ar_high = np.concatenate([ar_high, ar_high2], axis=1)
        ar_low = np.concatenate([ar_low, ar_low2], axis=1)
        conn_high = np.corrcoef(ar_high)
        conn_low = np.corrcoef(ar_low)

        plt.imshow(conn_high)
        plt.colorbar()
        plt.show()
        quit()
    quit()
        # fp_lsa_rl = fr'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSA\{sn}_rl_LSA.nii'
        # img_lsa_rl = image.load_img(fp_lsa_rl)
        # data_lsa_rl = img_lsa_rl.get_fdata()
def test_LSS_x_LSA():
    fp_LSS = r'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSS\100206_LR_LSS.nii'
    img_LSS = image.load_img(fp_LSS).get_fdata()
    fp_LSA = r'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSA\100206_LR_LSA.nii'
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


if __name__ == '__main__':
    # fp = r'H:\PycharmProjects_H\SchemeRep\HCP_gambling\LSS\100206_LR_LSS.nii'
    # img = image.load_img(fp)
    # print(img.shape)
    # LSS_gambling()
    make_conn()
    # unzip_all_gambling()
    # test_LSS_x_LSA()






