import logging
from functools import partial

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.image import high_variance_confounds
from tqdm import tqdm

from atlas_utils import get_atlas
from org_sns import get_sns
from utils import pickle_wrap, stdize


def load_rs_BOLD(fp='rs_medium', norm_std=False, combine_regions=False):

    if fp == 'rs_medium':
        f = partial(load_resting_data, medium=True, clean=True,
                    combine_regions=combine_regions)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=False,
                                                     f=f, YA_only=False)
    else:
        raise ValueError

    return sn_roi_act, sns, conn_trials


def get_sn_rs(sn, clean=True, compcor=True, light=False, medium=False,
              trad=False, near_OG=False, true_OG=False):
    fp_in = fr'fMRI_in/{sn}/resting/rs.nii.gz'
    img = image.load_img(fp_in)
    if true_OG:
        dir_mask = fr'H:\PycharmProjects_H\SchemeRep\fMRI_in\masks'
        fn_mask = fr'sub-{sn}_space-MNI152NLin2009cAsym_res-2_GrayMatter20.nii'
        fp_mask = fr'{dir_mask}\{fn_mask}'

        fp_in = fr'H:\PycharmProjects_H\SchemeRep\cache\confounds\{sn}_resting_confounds.tsv'
        df_confounds = pd.read_csv(fp_in, delimiter='\t')
        df_confounds = df_confounds.iloc[4:].reset_index(drop=True)
        img = image.index_img(img, slice(4, None))
        df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2))
        df_confounds = pd.concat([df_confounds, df_compcor], axis=1)
        # print('oooo')
        if int(sn) in [221, 222]:
            fp_mask = None
        img = image.clean_img(img, confounds=df_confounds, high_pass=1/128,
                              standardize=False, t_r=2, mask_img=fp_mask)
        return img.get_fdata()

    if near_OG: trad = True
    if clean:
        dir_mask = fr'H:\PycharmProjects_H\SchemeRep\fMRI_in\masks'
        fn_mask = fr'sub-{sn}_space-MNI152NLin2009cAsym_res-2_GrayMatter20.nii'
        fp_mask = fr'{dir_mask}\{fn_mask}'

        fp_in = fr'H:\PycharmProjects_H\SchemeRep\cache\confounds\{sn}_resting_confounds.tsv'
        df_confounds = pd.read_csv(fp_in, delimiter='\t')
        df_confounds = df_confounds.iloc[4:].reset_index(drop=True)
        img = image.index_img(img, slice(4, None))

        if light:
            # tcc = [f't_comp_cor_0{i}' for i in range(5)]
            # ccc = [f'c_comp_cor_0{i}' for i in range(5)]
            # wcc = [f'w_comp_cor_0{i}' for i in range(5)]
            # acc = [f'a_comp_cor_0{i}' for i in range(5)]
            # ecc = [f'edge_comp_0{i}' for i in range(5)]
            # ccs = set(tcc + ccc + wcc + acc + ecc)
            # f = lambda x: True if '_comp_' not in x else x in ccs
            # # print(df_confounds.columns)
            # keep_cols = [col for col in df_confounds.columns if f(col)]
            # # keep_cols = [col for col in keep_cols if 'cosine' not in col]
            # keep_cols = [col for col in keep_cols if 'global_' not in col]
            # keep_cols = [col for col in keep_cols if 'white_' not in col]
            # keep_cols = [col for col in keep_cols if 'csf_' not in col]
            #
            # df_confounds = df_confounds[keep_cols]
            # print(list(df_confounds.columns))

            cols = df_confounds.columns
            motion_outliers = [col for col in cols if 'motion_outlier' in col]
            # keep_cols = ['global_signal', 'global_signal_derivative1',
            #              'global_signal_derivative1_power2',
            #              'global_signal_power2', 'csf', 'csf_derivative1',
            #              'csf_power2', 'csf_derivative1_power2', 'white_matter',
            #              'white_matter_derivative1',
            #              'white_matter_derivative1_power2',
            #              'white_matter_power2']
            keep_cols  = ['trans_x', 'trans_x_derivative1', 'trans_y',
                          'trans_y_derivative1', 'trans_z',
                          'trans_z_derivative1', 'rot_x', 'rot_x_derivative1',
                          'rot_y', 'rot_y_derivative1', 'rot_z',
                          'rot_z_derivative1', ]
            # keep_cols += ['cosine00', 'cosine01', 'cosine02']
            # keep_cols += motion_outliers
            df_confounds = df_confounds[keep_cols]
            df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2,
                                                              mask_img=fp_mask))
            df_confounds = pd.concat([df_confounds, df_compcor], axis=1)
        elif medium:
            cols = df_confounds.columns
            keep_cols  = ['trans_x', 'trans_x_derivative1', 'trans_y',
                          'trans_y_derivative1', 'trans_z',
                          'trans_z_derivative1', 'rot_x', 'rot_x_derivative1',
                          'rot_y', 'rot_y_derivative1', 'rot_z',
                          'rot_z_derivative1', ]
            keep_cols += [col for col in cols if 't_comp_cor' in col]
            keep_cols += [col for col in cols if 'w_comp_cor' in col]
            keep_cols += [col for col in cols if 'c_comp_cor' in col]
            # keep_cols += [col for col in cols if 'edge_comp' in col]
            # keep_cols += [col for col in cols if 'motion_outlier' in col]
            # keep_cols += ['cosine00', 'cosine01', 'cosine02']

            # keep_cols = df_confounds.columns
            # keep_cols = [col for col in keep_cols if 'global_' not in col]
            # keep_cols = [col for col in keep_cols if 'white_' not in col]
            # keep_cols = [col for col in keep_cols if 'csf_' not in col]

            df_confounds = df_confounds[keep_cols]
        elif trad:
            cols = df_confounds.columns
            # motion_outliers = [col for col in cols if 'motion_outlier' in col]
            motion_outliers = []
            keep_cols = ['global_signal', 'global_signal_derivative1',
                         #'global_signal_derivative1_power2',
                         # 'global_signal_power2',
                         'csf', 'csf_derivative1',
                         # 'csf_power2', 'csf_derivative1_power2',
                         'white_matter',
                         'white_matter_derivative1',
                         # 'white_matter_derivative1_power2',
                         # 'white_matter_power2'
                         ]
            keep_cols += ['trans_x', 'trans_x_derivative1', 'trans_y',
                          'trans_y_derivative1', 'trans_z',
                          'trans_z_derivative1', 'rot_x', 'rot_x_derivative1',
                          'rot_y', 'rot_y_derivative1', 'rot_z',
                          'rot_z_derivative1', ]
            # keep_cols += ['cosine00', 'cosine01', 'cosine02']
            keep_cols += motion_outliers
            df_confounds = df_confounds[keep_cols]

        if near_OG:
            df_compcor = pd.DataFrame(high_variance_confounds(img, percentile=2,
                                                              mask_img=fp_mask))
            df_confounds = pd.concat([df_confounds, df_compcor], axis=1)

        if int(sn) in [221, 222]:
            fp_mask = None
        fp_mask = None # It appears that the GM masks disapeared 9/3/2024
        img = image.clean_img(img, confounds=df_confounds, high_pass=1/128,
                              standardize=False, t_r=2,
                              mask_img=fp_mask
                              )

    data = img.get_fdata()
    return data


def load_resting_data(raw_enc=False, lss_enc=False, lsa=False, YA_only=False,
                      sanity=False, combine_regions=False, sns_key='loose',
                      compcor=True, clean=True, light=False, medium=False,
                      trad=False, near_OG=False, true_OG=False):
    # print('test')
    # quit()
    # print(f'{sns_key=}')
    age2sn = get_sns(sns_key)
    # print(age2sn)
    # quit()
    if YA_only:
        sns = age2sn[1]
    else:
        sns = age2sn[1] + age2sn[2]

    # if (lss_enc and not lsa) or sanity:
    #     sns = sns[10:15]

    sn_roi_act = []
    atlas = get_atlas(combine_regions=combine_regions)
    bad_rs_sns = {'133'}
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    # sns = sns[::-1]

    for i, sn in tqdm(enumerate(sns), desc='Loading fMRI'):
        logging.debug(f'{sn=}')

        # if sanity:
        #     data = pickle_wrap(sanity_load, kwargs={'sn': sn})
        # else:

        data = pickle_wrap(get_sn_rs, kwargs={'sn': sn, 'clean': clean,
                                              'compcor': compcor,
                                              'light': light,
                                              'medium': medium,
                                              'trad': trad,
                                              'near_OG': near_OG,
                                              'true_OG': true_OG},
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
    # print(sn_roi_act.shape)
    sn_SD_trial = np.nanstd(sn_roi_act, axis=1)
    sn_SD_M = np.nanmean(sn_SD_trial, axis=1)
    sn_SD_trial_rel = sn_SD_trial / sn_SD_M[:, None]
    # print(sn_SD_trial_rel.shape)
    # print(sn_SD_trial_rel[12])
    sn_roi_act /= sn_SD_trial_rel[:, None, :]
    return sn_roi_act


def load_act_conn(norm_std, f=None, easy_override=False, YA_only=False,
                  RAM_cache=False):
    if f is None:
        f = load_resting_data
        # easy_override = True
    sn_roi_act, sns = pickle_wrap(f, easy_override=easy_override,
                                  kwargs={'YA_only': YA_only},
                                  RAM_cache=RAM_cache)

    # sn_roi_act = sn_roi_act[:, :, 4:] # bad trials to start?
    bad_rs_sns = {'133'}
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    # print(f'{sn_roi_act.shape=}')
    sn_roi_act = stdize(sn_roi_act, axis=2, nans=True)
    assert len(sn_roi_act.shape) == 3, f'More than 3 dims: {sn_roi_act.shape=}'
    if norm_std: sn_roi_act = normalize_std_over_time(sn_roi_act)
    conn_trials = sn_roi_act[..., None, :] * \
                  sn_roi_act[..., None, :, :]
    return sn_roi_act, sns, conn_trials
