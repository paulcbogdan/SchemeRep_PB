from functools import partial

import numpy as np
import pandas as pd
import scipy.stats as stats
from nilearn import image
from tqdm import tqdm

from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap


def load_rs_BOLD(norm_std=False, combine_regions=False):
    f = partial(load_resting_data,
                combine_regions=combine_regions)
    sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                 easy_override=False,
                                                 f=f, YA_only=False)
    return sn_roi_act, sns, conn_trials


def get_sn_rs(sn):
    fp_in = fr'fMRI_in/{sn}/resting/rs.nii.gz'
    img = image.load_img(fp_in)

    fp_in = fr'cache\confounds\{sn}_resting_confounds.tsv'
    df_confounds = pd.read_csv(fp_in, delimiter='\t')
    df_confounds = df_confounds.iloc[4:].reset_index(drop=True)
    img = image.index_img(img, slice(4, None))

    cols = df_confounds.columns
    keep_cols = ['trans_x', 'trans_x_derivative1', 'trans_y',
                 'trans_y_derivative1', 'trans_z',
                 'trans_z_derivative1', 'rot_x', 'rot_x_derivative1',
                 'rot_y', 'rot_y_derivative1', 'rot_z',
                 'rot_z_derivative1', ]
    keep_cols += [col for col in cols if 't_comp_cor' in col]
    keep_cols += [col for col in cols if 'w_comp_cor' in col]
    keep_cols += [col for col in cols if 'c_comp_cor' in col]
    df_confounds = df_confounds[keep_cols]

    if int(sn) in [221, 222]:
        fp_mask = None
        # No mask available for these subjects.
        # This just means that the analysis will be a bit slower since it is run on all voxels

    img = image.clean_img(img, confounds=df_confounds, high_pass=1 / 128,
                          standardize=False, t_r=2, mask_img=fp_mask)

    data = img.get_fdata()
    return data


def load_resting_data(YA_only=False, combine_regions=False, ):
    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
           '113', '114', '115', '116', '117', '118', '119', '120', '123', '124', '125',
           '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136',
           '137', '138', '201', '202', '203', '204', '205', '206', '207', '208', '209',
           '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '221',
           '222', '224', '225', '227', '230', '231', '232', '233', '234', '235', '239']

    if YA_only:
        sns = [sn for sn in sns if int(sn) < 200]

    sn_roi_act = []
    atlas = get_atlas(combine_regions=combine_regions)
    bad_rs_sns = {'133'}
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']

    for i, sn in tqdm(enumerate(sns), desc='Loading fMRI'):
        data = pickle_wrap(get_sn_rs, kwargs={'sn': sn},
                           easy_override=False)
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
    sn_roi_act = np.array(sn_roi_act)
    return sn_roi_act, sns


def normalize_std_over_time(sn_roi_act):
    sn_SD_trial = np.nanstd(sn_roi_act, axis=1)
    sn_SD_M = np.nanmean(sn_SD_trial, axis=1)
    sn_SD_trial_rel = sn_SD_trial / sn_SD_M[:, None]
    sn_roi_act /= sn_SD_trial_rel[:, None, :]
    return sn_roi_act


def load_act_conn(norm_std, f=None, easy_override=False, YA_only=False,
                  RAM_cache=False):
    if f is None:
        f = load_resting_data
    sn_roi_act, sns = pickle_wrap(f, easy_override=easy_override,
                                  kwargs={'YA_only': YA_only},
                                  RAM_cache=RAM_cache)

    bad_rs_sns = {'133'}
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    sn_roi_act = stats.zscore(sn_roi_act, axis=2, nan_policy='omit')
    assert len(sn_roi_act.shape) == 3, f'More than 3 dims: {sn_roi_act.shape=}'
    if norm_std: sn_roi_act = normalize_std_over_time(sn_roi_act)
    conn_trials = sn_roi_act[..., None, :] * \
                  sn_roi_act[..., None, :, :]
    return sn_roi_act, sns, conn_trials
