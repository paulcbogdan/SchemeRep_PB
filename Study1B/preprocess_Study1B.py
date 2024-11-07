import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import os
from collections import defaultdict
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
import numpy as np
from nilearn import image

import time


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
    if not cont_PE:
        df_trials['trial_type'] = df_trials['same'].apply(
            lambda x: 'low_PE' if x else 'high_PE')
    df_trials.reset_index(drop=True, inplace=True)
    df_trials['block_num'] = df_trials.index // 8
    df_trials['trials_all'] = list(range(len(df_trials)))
    return df_trials


def get_df_PE(sn, RL_LR, learning_rate=1,
              drop_first=False, reset_trial0=False,
              for_prepoc=False):
    if RL_LR == 'both':
        df_trials_LR = get_df_events(sn, 'LR', cont_PE=learning_rate)
        df_trials_LR['variant'] = 'LR'
        df_trials = get_df_events(sn, 'RL', cont_PE=learning_rate)
        df_trials['variant'] = 'RL'
        df_trials = pd.concat([df_trials, df_trials_LR], ignore_index=True)
    else:
        df_trials = get_df_events(sn, RL_LR, cont_PE=learning_rate)

    if for_prepoc:
        df_trials['trial_type'] = 'NA'
        return df_trials

    PE_l = []
    E = 0
    learning = learning_rate
    for idx, row in df_trials.iterrows():
        if row['trials_all'] == 0 and reset_trial0:
            E = 0
        if row['event'] == 'win':
            PE = 1 - E
            E = E * (1 - learning) + learning
        elif row['event'] == 'loss':
            PE = -1 - E
            E = E * (1 - learning) - learning
        else:
            PE = 0 - E
            E = E * (1 - learning)
        PE_l.append(PE)

    df_trials['PE'] = PE_l
    df_trials['PE_abs'] = df_trials['PE'].abs()

    if drop_first:
        if RL_LR == 'both':
            df_trials['trial_within_block'] = list(range(8)) * 8
        else:
            df_trials['trial_within_block'] = list(range(8)) * 4
        df_trials.loc[df_trials['trial_within_block'] == 0, 'PE_abs'] = np.nan

    for event in ['loss', 'win', 'neut']:
        df_event = df_trials[df_trials['event'] == event]
        med_PE = df_event['PE_abs'].median()
        df_trials.loc[df_trials['event'] == event, 'trial_type'] = df_trials.loc[
            df_trials['event'] == event, 'PE_abs'].apply(
            lambda x: 'low_PE' if x < med_PE else 'high_PE' if ~pd.isna(x) else 'dropped')

    pd.set_option('display.max_rows', 2000)

    df_trials.drop(columns=['same'], inplace=True)

    if drop_first:
        if RL_LR == 'both':
            df_trials['trial_within_block'] = list(range(8)) * 8
        else:
            df_trials['trial_within_block'] = list(range(8)) * 4
        df_trials.loc[df_trials['trial_within_block'] == 0, 'trial_type'] = 'dropped'

    # code just below accounts for odd numbers of loss and/or win trials.
    #   drops last trials as needed
    df_lw = df_trials[df_trials['event'] != 'neut']

    df_loss = df_lw[df_lw['event'] == 'loss']
    df_loss_low = df_loss[df_loss['trial_type'] == 'low_PE']
    df_loss_high = df_loss[df_loss['trial_type'] == 'high_PE']
    if len(df_loss_low) > len(df_loss_high):
        df_trials.loc[df_loss_low.index[-1], 'trial_type'] = 'drop'
    elif len(df_loss_low) < len(df_loss_high):
        df_trials.loc[df_loss_high.index[-1], 'trial_type'] = 'drop'
    df_win = df_lw[df_lw['event'] == 'win']
    df_win_low = df_win[df_win['trial_type'] == 'low_PE']
    df_win_high = df_win[df_win['trial_type'] == 'high_PE']
    if len(df_win_low) > len(df_win_high):
        df_trials.loc[df_win_low.index[-1], 'trial_type'] = 'drop'
    elif len(df_win_low) < len(df_win_high):
        df_trials.loc[df_win_high.index[-1], 'trial_type'] = 'drop'

    df_lw = df_trials[df_trials['event'] != 'neut']
    df_low = df_lw[df_lw['trial_type'] == 'low_PE']
    df_high = df_lw[df_lw['trial_type'] == 'high_PE']
    assert len(df_low) == len(df_high), f'{len(df_low)=}, {len(df_high)=}'

    if RL_LR == 'both':
        df_trials_RL = df_trials[df_trials['variant'] == 'RL']
        df_trials_LR = df_trials[df_trials['variant'] == 'LR']
        return df_trials_RL, df_trials_LR
    else:
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


def load_motion(sn, lr, rs=False, drive='F'):
    if rs:
        fp_motion = fr'{drive}:\HCP_RS_unzipped\{sn}\MNINonLinear\Results\rfMRI_REST1_{lr}\Movement_Regressors.txt'
    else:
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
        condition_counter[trial_condition] += 1
        trial_name = f"{trial_condition}__{condition_counter[trial_condition]:03d}"
        df_trials.loc[i_trial, "trial_type"] = trial_name

    frame_times = np.linspace(0, 192, 253, endpoint=False)
    df_confounds = load_motion(sn, lr, rs=False)

    fp_mask = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\brainmask_fs.2.nii.gz'
    img = image.load_img(img)
    mask = image.load_img(fp_mask)
    global_signal = img.get_fdata()[mask.get_fdata() > 0].mean(axis=0)
    df_confounds['global'] = global_signal

    X1 = make_first_level_design_matrix(
        frame_times,
        df_trials,
        add_regs=df_confounds,
        hrf_model='spm',
    )

    glm = FirstLevelModel(slice_time_ref=0.5, t_r=192 / 253,
                          signal_scaling=(0, 1), high_pass=1 / 128,
                          minimize_memory=True,
                          mask_img=fp_mask,
                          verbose=100)

    print('Fitting LSA model...')
    glm = glm.fit(img, design_matrices=X1)

    del X1
    betas = []
    print('Making contrasts...')
    for i_trial, trial in df_trials.iterrows():
        trial_name = trial["trial_type"]
        beta_map = glm.compute_contrast(trial_name,
                                        output_type='effect_size')
        beta = beta_map.get_fdata()[..., 0]
        del beta_map
        betas.append(deepcopy(beta))

    betas = np.array(betas)
    betas = np.transpose(betas, (1, 2, 3, 0))

    beta_img = image.new_img_like(img, betas)

    fp_lsa = fr'HCP_gambling\LSA\{sn}_{lr}_LSA_global_nocc.nii'
    print(f'Out: {fp_lsa=}')
    beta_img.to_filename(fp_lsa)


def do_LSS(img, df_trials, sn, lr):
    df_trials.reset_index(drop=True, inplace=True)
    betas = []
    df_confounds = load_motion(sn, lr, rs=False)

    fp_mask = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_{lr}\brainmask_fs.2.nii.gz'
    img = image.load_img(img)
    mask = image.load_img(fp_mask)
    global_signal = img.get_fdata()[mask.get_fdata() > 0].mean(axis=0)
    df_confounds['global'] = global_signal

    for i in tqdm(range(len(df_trials)), desc=f'Cooking LSS: {sn} ({lr})',
                  position=0, leave=True):
        df_trial, i_name = lss_transformer(df_trials, i)

        frame_times = np.linspace(0, 192, 253, endpoint=False)

        X1 = make_first_level_design_matrix(
            frame_times,
            df_trial,
            add_regs=df_confounds,
            hrf_model='spm',
        )
        glm = FirstLevelModel(slice_time_ref=0.5, t_r=192 / 253,
                              signal_scaling=(0, 1), high_pass=1 / 128,
                              minimize_memory=True,
                              mask_img=fp_mask,
                              verbose=100)
        glm = glm.fit(img, design_matrices=X1)
        beta_map = glm.compute_contrast(i_name,
                                        output_type='effect_size')
        del glm
        beta = beta_map.get_fdata()[..., 0]
        del beta_map
        del X1
        betas.append(deepcopy(beta))

    betas = np.array(betas)
    betas = np.transpose(betas, (1, 2, 3, 0))

    beta_img = image.new_img_like(img, betas)

    fp_lss = fr'HCP_gambling\LSS\{sn}_{lr}_LSS_global_nocc.nii'
    beta_img.to_filename(fp_lss)
    print(f'Out: {fp_lss=}')


def LSA_LSS_gambling(lsa=True, easy_override=False, ):
    sns = os.listdir(r'G:\HCP_gambling')
    sns = list(sns)
    sns = sorted(sns)

    bad_sns = []
    for sn in sns:
        try:
            fp_img = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_LR\tfMRI_GAMBLING_LR.nii.gz'
            if lsa:
                fp_lsa_lr = fr'HCP_gambling\LSA\{sn}_lr_LSA_global_nocc.nii'
                if not os.path.exists(fp_lsa_lr) or easy_override:
                    df_events_LR = get_df_PE(sn, 'LR', for_prepoc=True)
                    do_LSA(fp_img, df_events_LR, sn, 'LR')
            else:
                fp_lss_lr = fr'HCP_gambling\LSS\{sn}_lr_LSS_global_nocc.nii'
                if not os.path.exists(fp_lss_lr) or easy_override:
                    df_events_LR = get_df_PE(sn, 'LR', for_prepoc=True)
                    do_LSS(fp_img, df_events_LR, sn, 'LR')

            fp_img = fr'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\tfMRI_GAMBLING_RL.nii.gz'
            if lsa:
                fp_lsa_rl = fr'HCP_gambling\LSA\{sn}_rl_LSA_global_nocc.nii'
                if not os.path.exists(fp_lsa_rl) or easy_override:
                    df_events_RL = get_df_PE(sn, 'RL', for_prepoc=True)
                    do_LSA(fp_img, df_events_RL, sn, 'RL')
            else:
                fp_lss_rl = fr'HCP_gambling\LSS\{sn}_rl_LSS_global_nocc.nii'
                if not os.path.exists(fp_lss_rl) or easy_override:
                    df_events_RL = get_df_PE(sn, 'RL', for_prepoc=True)
                    do_LSS(fp_img, df_events_RL, sn, 'RL')
            print(f'Done: {sn}')

        except ValueError as e:
            print(f'ValueError: {sn}, {e=}')
            bad_sns.append(sn)
        except Exception as e:
            print(f'ERROR: {sn=}, {e=}')
            time.sleep(1)
            bad_sns.append(sn)


if __name__ == '__main__':
    LSA_LSS_gambling()
