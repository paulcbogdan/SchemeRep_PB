import os

from nilearn.plotting import plot_design_matrix

os.chdir(r'E:\PycharmProjects_E\SchemeRep')
from collections import defaultdict

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.image import high_variance_confounds
from tqdm import tqdm

from org_sns import get_sns
from utils import HCP_CACHE, HCP_ROOT
import matplotlib.pyplot as plt
from copy import deepcopy

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


def pad_vector(contrast_, n_columns):
    """Append zeros in contrast vectors."""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


def block2trials(df, skip_rate=4):
    df_new_as_l = []
    for i, row in df.iterrows():
        for j in range(int(row['duration'])):
            if j % skip_rate != 0:
                continue
            row_new = row.copy()
            row_new['onset'] = row['onset'] + j
            row_new['duration'] = skip_rate
            df_new_as_l.append(row_new)
    df_new = pd.DataFrame(df_new_as_l)
    df_new.reset_index(drop=True, inplace=True)
    return df_new


def get_LSS_img(sn, run, only_2bk=True, easy_override=False, skip_rate=4,
                hcp=True, lsa=False):
    if hcp:
        lsa_str = 'lsa' if lsa else 'lss'
        fp_lss = (fr'{HCP_CACHE}_lss\{sn}_{run}_{only_2bk}_{skip_rate}_'
                  fr'{lsa_str}.nii')
    else:
        lsa_str = 'lsa' if lsa else 'lss'
        fp_lss = fr'cache\pb_LSS\{sn}_r{run}_{lsa_str}.nii'
    if os.path.isfile(fp_lss) and not easy_override:
        img = image.load_img(fp_lss)
        data = img.get_fdata()
        return data

    if hcp:
        subj_dir = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results'
        fp_in = fr'{subj_dir}\tfMRI_WM_{run}\tfMRI_WM_{run}.nii'
        img = image.load_img(fp_in)

        df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2))
        df_motion = load_motion(sn, run) # TODO: compcor not used previously
        df_confounds = pd.concat([df_compcor, df_motion], axis=1)
        fp_0bk = fr'{subj_dir}\tfMRI_WM_{run}\EVs\0bk.txt'
        df_0bk = pd.read_csv(fp_0bk, sep=' ',
                             names=['onset', 'duration', 'uncertain'])
        df_0bk['trial_type'] = '0bk'
        fp_2bk = fr'{subj_dir}\tfMRI_WM_{run}\EVs\2bk.txt'
        df_2bk = pd.read_csv(fp_2bk, sep=' ',
                                names=['onset', 'duration', 'uncertain'])
        df_2bk['trial_type'] = '2bk'
        df_events = pd.concat([df_0bk, df_2bk], axis=0)
        if only_2bk:
            df_events = df_events.loc[df_events['trial_type'] == '2bk']
        df_trials = block2trials(df_events, skip_rate=skip_rate)
        frame_times = np.arange(405) * .72
        num_vols = img.shape[-1]
        sample_masks = list(range(num_vols))
    else:
        df_all = pd.read_csv(r'cache/trial_info_ENC.csv')
        df_sn = df_all[df_all['Subject'] == int(sn)]
        df_trials = df_sn[df_sn['Run'] == run]
        df_trials = df_trials[['OnsetObj']]
        df_trials.rename(columns={'OnsetObj': 'onset'}, inplace=True)
        df_trials['duration'] = 1
        df_trials['trial_type'] = 'obj'
        fp_in = fr'G:\SchemeRep_raw_data_dir_preproc\{sn}\ENC\BOLD_run{run}.nii'
        img = image.load_img(fp_in)

        fp_confounds = rf'cache\confounds\102_2_{run}_confounds.tsv'
        if not os.path.isfile(fp_confounds):
            print(f'No confounds found ({sn}): {fp_confounds=}')
            df_confounds = pd.DataFrame(high_variance_confounds(img, percentile=2))
        else:
            df_confounds = pd.read_csv(fp_confounds, delimiter='\t')

        nuisance_cols = ["global_signal", "white_matter", "csf",
                         "dvars", "framewise_displacement", "rmsd",
                         "trans_x", "trans_y", "trans_z", "rot_x",
                         "rot_y", "rot_z"]
        df_confounds = df_confounds[nuisance_cols]

        frame_times = np.arange(img.shape[-1]) * 2
        num_vols = img.shape[-1]
        sample_masks = list(range(2, num_vols))
    sample_masks = np.array(sample_masks)
    df_trials.reset_index(drop=True, inplace=True)
    num_nans = df_confounds.isna().sum().sum()
    # print(f'Number of nans in df_confounds: {num_nans=}')
    # print(f'{df_confounds.isna()=}')
    # for col in df_confounds.columns:
    #     num_nans = df_confounds[col].isna().sum()
    #     print(f'{col=}, {num_nans=}')


    assert num_nans < 10, f'df_confounds: {num_nans=}'
    df_confounds.fillna(0, inplace=True)

    # print(len(df_trials))
    # quit()
    # if not hcp:
    #     assert len(df_trials) == 38, f'too many trials: {len(df_trials)=}'
    # return
    # print(f'{len(df_trials)=}')
    # quit()
    if lsa:
        condition_counter = defaultdict(lambda: 0)
        for i_trial, trial in df_trials.iterrows():
            trial_condition = trial["trial_type"]
            condition_counter[trial_condition] += 1
            trial_name = f"{trial_condition}__{condition_counter[trial_condition]:03d}"
            df_trials.loc[i_trial, "trial_type"] = trial_name

        X1 = make_first_level_design_matrix(
            frame_times,
            df_trials,
            drift_model="polynomial",
            drift_order=3,
            add_regs=df_confounds,
            hrf_model='glover',
        )
        # plot_design_matrix(X1)
        # plt.show()
        # quiT()
        glm = FirstLevelModel()
        glm = glm.fit(img, design_matrices=X1, sample_masks=sample_masks)
        del X1
        betas = []
        for i_trial, trial in df_trials.iterrows():
            trial_name = trial["trial_type"]
            # print(f'Commute contrast: {i_trial} | {trial_name}')
            beta_map = glm.compute_contrast(trial_name,
                                            output_type='effect_size')
            # betas.append(beta_map.get_fdata()[..., 0])

            beta = beta_map.get_fdata()[..., 0]
            del beta_map
            betas.append(deepcopy(beta))
    else:
        betas = []
        for i in tqdm(range(len(df_trials)), desc=f'Cooking LSS: {sn}',
                      position=0, leave=True):
            print(f'{i=}')
            df_trial, i_name = lss_transformer(df_trials, i)

            X1 = make_first_level_design_matrix(
                frame_times,
                df_trial,
                drift_model="polynomial",
                drift_order=3,
                add_regs=df_confounds,
                hrf_model='glover',
            )
            glm = FirstLevelModel(memory=r'cache\nilearn_memory',
                                  memory_level=1)
            glm = glm.fit(img, design_matrices=X1, sample_masks=sample_masks)
            beta_map = glm.compute_contrast(i_name,
                                            output_type='effect_size')
            del glm
            beta = beta_map.get_fdata()[..., 0]
            del beta_map
            del X1
            betas.append(deepcopy(beta))

    betas = np.array(betas)
    betas = np.transpose(betas, (1, 2, 3, 0))
    # print(f'{betas.shape=}')
    # quit()
    beta_img = image.new_img_like(img, betas)
    # beta_img = image.concat_imgs(betas)
    beta_img.to_filename(fp_lss)
    data = beta_img.get_fdata()
    print(f'{data.shape=}')
    return data


def load_motion(sn, lr):
    subj_dir = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results'
    fp_motion = fr'{subj_dir}\tfMRI_WM_{lr}\Movement_Regressors.txt'
    motion = np.loadtxt(fp_motion)[:, :6]
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    df = pd.DataFrame(motion, columns=add_reg_names)
    return df

if __name__ == '__main__':
    run = 1
    fp =  rf'cache\confounds\104_2_{run}_confounds.tsv'
    df = pd.read_csv(fp, delimiter='\t')
    print(plt.plot(df['global_signal']))
    plt.show()
    # print(df)
    quit()

    age2sns = get_sns()
    sns = age2sns[1] + age2sns[2]
    for sn in sns:
        print(f'{sn=}')
        if sn in ['116', '125', '135']: continue
        try:
            get_LSS_img(sn, 1, hcp=False, lsa=True, easy_override=False)
        except ValueError as e:
            print(f'({sn}), {e=}')

