import os

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.image import high_variance_confounds
from tqdm import tqdm

from utils import HCP_CACHE, HCP_ROOT


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
                hcp=True):
    if hcp:
        fp_lss = fr'{HCP_CACHE}_lss\{sn}_{run}_{only_2bk}_{skip_rate}.nii'
    else:
        fp_lss = fr'cache\pb_LSS\{sn}_{run}.nii'
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
        df_confounds = pd.DataFrame(high_variance_confounds(img, percentile=2))
        frame_times = np.arange(img.shape[-1]) * 2

    betas = []
    for i in tqdm(range(len(df_trials)), desc=f'Cooking LSS: {sn}',
                  position=0, leave=True):
        df_trial, i_name = lss_transformer(df_trials, i)

        X1 = make_first_level_design_matrix(
            frame_times,
            df_trial,
            drift_model="polynomial",
            drift_order=3,
            add_regs=df_confounds,
            hrf_model='glover',
        )
        glm = FirstLevelModel()
        glm = glm.fit(img, design_matrices=X1)
        beta_map = glm.compute_contrast(i_name, output_type='effect_size')
        # design_matrix = glm.design_matrices_[0]
        # plot_design_matrix(design_matrix)
        # plt.show()
        betas.append(beta_map)
    #     print(beta_map.shape)
    #     quit()
    # betas = np.array(betas)
    beta_img = image.concat_imgs(betas)
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
