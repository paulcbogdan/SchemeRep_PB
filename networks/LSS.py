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
import psutil
import logging
from scipy import io

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


def get_LSS_img(sn, run, sess=2, only_2bk=True, easy_override=False,
                skip_rate=4, hcp=True, lsa=False):

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
    process = psutil.Process()

    rand_int = np.random.randint(0, 10000)
    fp_log = fr"E:\PycharmProjects_E\SchemeRep\pb_crash_log_{rand_int}.log"
    logging.basicConfig(filemode='w',
        filename=fp_log,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.DEBUG,)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)
    logging.debug(f'Running LSS: {sn=}, {run=}, {hcp=}, {lsa=}')
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
        fp_mask = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results\tfMRI_WM_LR\brainmask_fs.2.nii.gz'

        # num_vols = img.shape[-1]
        # sample_masks = list(range(num_vols))
    else:
        dir_mask = fr'E:\PycharmProjects_E\SchemeRep\fMRI_in\masks'
        fn_mask = fr'sub-{sn}_space-MNI152NLin2009cAsym_res-2_GrayMatter20.nii'
        fp_mask = fr'{dir_mask}\{fn_mask}'
        df_all = pd.read_csv(r'cache/trial_info_ENC.csv')
        df_sn = df_all[df_all['Subject'] == int(sn)]

        df_trials = df_sn[df_sn['Run'] == run]
        df_copy = df_trials.copy()
        df_copy['onset'] = df_copy['OnsetScene']
        # df_copy['onset'] -= 2
        df_trials['onset'] = df_trials['OnsetObj']
        # df_trials['onset'] += 2
        df_copy['duration'] = 3
        df_copy = df_copy[['onset', 'duration']]
        df_copy['trial_type'] = 'scn'
        # df_copy['trial_type'] = 'obj'
        df_trials['duration'] = 4
        df_trials = df_trials[['onset', 'duration']]
        df_trials['trial_type'] = 'obj'
        df_trials = pd.concat([df_trials, df_copy], axis=0)
        # df_trials = df_trials[['OnsetObj']]
        # df_trials.rename(columns={'OnsetObj': 'onset'}, inplace=True)
        # df_trials['duration'] = 4
        sess2name = {1: 'BL', 2: 'ENC'}
        sess_name = sess2name[sess]
        fp_in = fr'E:\PycharmProjects_E\SchemeRep\fMRI_in_BOLD\{sn}\{sess_name}\BOLD_run{run}.nii.gz'
        MBs = process.memory_info().rss / 1024 / 1024
        logging.debug(f'Loading BOLD image: {MBs=:.2f}')
        img = image.load_img(fp_in)

        img = image.index_img(img, list(range(4, img.shape[-1])))
        frame_times = np.arange(img.shape[-1]) * 2

        dir_mat = r'E:\PycharmProjects_E\SchemeRep\nuisance_regressors'
        fp_mat = fr'{dir_mat}\sub-{sn}_ses-{sess}_task-{sess_name}_run-0{run}_' \
                    'desc-confounds_timeseries_use_univ.mat'

        fp_confounds = rf'cache\confounds\{sn}_{sess}_{run}_confounds.tsv'
        if os.path.isfile(fp_confounds):
            df_confounds = pd.read_csv(fp_confounds, delimiter='\t')
            df_confounds = df_confounds.iloc[4:] #
            confounds = ["global_signal", "white_matter", "csf",
                         "dvars", "framewise_displacement", "rmsd",
                         "trans_x", "trans_y", "trans_z", "rot_x",
                         "rot_y", "rot_z"]
            df_confounds = df_confounds[confounds]
        elif False and os.path.isfile(fp_mat):
            mat_enc = io.loadmat(fp_mat)
            names = [name[0] for name in mat_enc['names'][0]]
            df_confounds = pd.DataFrame(mat_enc['R'], columns=names)
        else:
            print(f'No confounds found ({sn}): {fp_confounds=}')
            df_confounds = pd.DataFrame(high_variance_confounds(img,
                                                                percentile=2))

        # sample_masks = list(range(2, num_vols))
    # sample_masks = np.array(sample_masks)
    df_trials.reset_index(drop=True, inplace=True)
    num_nans = df_confounds.isna().sum().sum()

    # print(f'Number of nans in df_confounds: {num_nans=}')
    # print(f'{df_confounds.isna()=}')
    # for col in df_confounds.columns:
    #     num_nans = df_confounds[col].isna().sum()
    #     print(f'{col=}, {num_nans=}')
    # print(f'{len(df_trials)=}')
    # quit()


    assert num_nans < 10, f'df_confounds: {num_nans=}'
    df_confounds.fillna(0, inplace=True)
    print(f'Running beta-models: {lsa=}')



    logging.debug(f'Running beta-models ({sn}): {lsa=}')
    # print(img.shape)
    # quit()

    # print(len(df_trials))
    # quit()
    # if not hcp:
    #     assert len(df_trials) == 38, f'too many trials: {len(df_trials)=}'
    # return
    # print(f'{len(df_trials)=}')
    # quit()

    # log = open(r"E:\PycharmProjects_E\SchemeRep\pb_crash_log.log", "a")


    if lsa:
        condition_counter = defaultdict(lambda: 0)
        for i_trial, trial in df_trials.iterrows():
            trial_condition = trial["trial_type"]
            if 'scn' in trial_condition: continue

            condition_counter[trial_condition] += 1
            trial_name = f"{trial_condition}__{condition_counter[trial_condition]:03d}"
            df_trials.loc[i_trial, "trial_type"] = trial_name
        # print(df_trials['trial_type'].value_counts())
        # quit()

        MBs = process.memory_info().rss / 1024 / 1024
        logging.debug(f'Making design matrix: {MBs=:.2f}')
        X1 = make_first_level_design_matrix(
            frame_times,
            df_trials,
            # drift_model="polynomial",
            # drift_order=3,
            add_regs=df_confounds,
            hrf_model='spm + derivative + dispersion',  #
        )
        # plot_design_matrix(X1)
        # plt.show()
        # quit()

        MBs = process.memory_info().rss / 1024 / 1024
        logging.debug(f'Making first level model: {MBs=:.2f}')
        glm = FirstLevelModel(slice_time_ref=0.5, t_r=.72 if hcp else 2.0,
                              signal_scaling=(0, 1), high_pass=1/128,
                              minimize_memory=True,
                              mask_img=fp_mask,
                              verbose=100)
        MBs = process.memory_info().rss / 1024 / 1024
        logging.debug(f'Fitting first level model: {MBs=:.2f}')
        glm = glm.fit(img, design_matrices=X1)
        MBs = process.memory_info().rss / 1024 / 1024
        logging.debug(f'Succesfully fit: {MBs=:.2f}')
        del X1
        betas = []
        for i_trial, trial in df_trials.iterrows():
            trial_name = trial["trial_type"]
            if 'obj__' not in trial_name:
                continue
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Onto trial {i_trial}, {trial}: {MBs=:.2f}')

            # print(f'Commute contrast: {i_trial} | {trial_name}')
            beta_map = glm.compute_contrast(trial_name,
                                            output_type='effect_size')

            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Getting beta map fdata {beta_map.shape=}: {MBs=:.2f}')
            beta = beta_map.get_fdata()[..., 0]
            del beta_map
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Making a deep copy: {MBs=:.2f}')
            betas.append(deepcopy(beta))
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Appended beta deepcopy: {MBs=:.2f}')
    else:
        betas = []
        for i in tqdm(range(len(df_trials)), desc=f'Cooking LSS: {sn}',
                      position=0, leave=True):

            df_trial, i_name = lss_transformer(df_trials, i)
            print(f'{i_name=}')
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Onto trial {i}, {i_name}: {MBs=:.2f}')
            if ('obj__' not in i_name) and ('2bk__' not in i_name):
                continue
            # print('test')
            # quit()
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Making design matrix: {MBs=:.2f}')
            X1 = make_first_level_design_matrix(
                frame_times,
                df_trial,
                # drift_model="polynomial",
                # drift_order=3,
                add_regs=df_confounds,
                hrf_model='spm + derivative + dispersion', #
            )
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Making first level model: {MBs=:.2f}')
            glm = FirstLevelModel(slice_time_ref=0.5, t_r=.72 if hcp else 2.0,
                                  signal_scaling=(0, 1), high_pass=1/128,
                                  minimize_memory=True,
                                  mask_img=fp_mask,
                                  verbose=100)
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Fitting first level model: {MBs=:.2f}')
            glm = glm.fit(img, design_matrices=X1)
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Computing contrast: {MBs=:.2f}')
            beta_map = glm.compute_contrast(i_name,
                                            output_type='effect_size')
            del glm
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Getting beta map fdata {beta_map.shape=}: {MBs=:.2f}')
            beta = beta_map.get_fdata()[..., 0]
            del beta_map
            del X1
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Making a deep copy: {MBs=:.2f}')
            betas.append(deepcopy(beta))
            MBs = process.memory_info().rss / 1024 / 1024
            logging.debug(f'Appended beta deepcopy: {MBs=:.2f}')
    MBs = process.memory_info().rss / 1024 / 1024
    logging.debug(f'Done! {MBs=:.2f}')

    betas = np.array(betas)
    betas = np.transpose(betas, (1, 2, 3, 0))
    # print(f'{betas.shape=}')
    # quit()
    beta_img = image.new_img_like(img, betas)
    # beta_img = image.concat_imgs(betas)
    beta_img.to_filename(fp_lss)
    MBs = process.memory_info().rss / 1024 / 1024
    logging.debug(f'saved: {MBs=:.2f}')
    data = beta_img.get_fdata()
    print(f'Betas img shape: {data.shape}')
    return data


def load_motion(sn, lr):
    subj_dir = fr'{HCP_ROOT}\{sn}\MNINonLinear\Results'
    fp_motion = fr'{subj_dir}\tfMRI_WM_{lr}\Movement_Regressors.txt'
    motion = np.loadtxt(fp_motion)[:, :12]
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz",
                     "dtx", "dty", "dtz", "drx", "dry", "drz"]
    df = pd.DataFrame(motion, columns=add_reg_names)
    return df

if __name__ == '__main__':
    # run = 1
    # fp =  rf'cache\confounds\104_2_{run}_confounds.tsv'
    # df = pd.read_csv(fp, delimiter='\t')
    # print(plt.plot(df['global_signal']))
    # plt.show()
    # # print(df)
    # quit()

    age2sns = get_sns()
    sns = age2sns[1] + age2sns[2]
    for sn in sns:
        print(f'{sn=}')
        if sn in ['116', '125', '135']: continue

        try:
            # get_LSS_img(sn, 1, hcp=False, lsa=False, easy_override=False)
            get_LSS_img(sn, 1, hcp=False, lsa=True, easy_override=False)
        except ValueError as e:
            print(f'({sn}), {e=}')

