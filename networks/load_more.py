import logging
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.image import high_variance_confounds
from scipy import stats as stats
from tqdm import tqdm

from LSS import get_LSS_img
from atlas_utils import get_atlas
from old.modularity import get_partition_cross
from old.network_funcs import load_FC_for_Lifu
from org_sns import get_sns
from organize_bhv import get_trial_info
from utils import pickle_wrap, timing, stdize, load_ni_w_nan_fps
from vendor_partitioning import get_vendor_partitions


def load_a(fp='pb_lss', norm_std=False, f=None,):

    if f is not None:
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=False)
    elif fp == 'sanity':
        f = partial(load_resting_data, YA_only=False, sanity=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=True)
    elif fp == 'raw_enc':
        f = partial(load_resting_data, raw_enc=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=False)
    elif fp == 'pb_lsa':
        f = partial(load_resting_data, raw_enc=False, lss_enc=True, lsa=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=True)
    elif fp == 'pb_lss':
        f = partial(load_resting_data, raw_enc=False, lss_enc=True, lsa=False,
                    YA_only=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                     easy_override=True)
    elif fp == 'rs':
        print('load act conn')
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     )
    elif fp == 'rs_noclean':
        f = partial(load_resting_data, compcor=False, clean=False)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     f=f, YA_only=False)
    elif fp == 'rs_nocc':
        f = partial(load_resting_data, compcor=False)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     f=f, YA_only=False)
    elif fp == 'rs_light':
        f = partial(load_resting_data, compcor=True, light=True, clean=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=True,
                                                     f=f, YA_only=True)
    elif fp == 'rs_medium':
        f = partial(load_resting_data, medium=True, clean=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=False,
                                                     f=f, YA_only=False)
    elif fp == 'rs_trad':
        f = partial(load_resting_data, clean=True, trad=True)
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=False,
                                                     f=f, YA_only=True)
    else:
        sn_roi_act, conn_trials, sns = \
            pickle_wrap(get_dfs_conn_trials, kwargs={'fp': fp, 'single': False,
                                                     'w_activity': True,
                                                     'squeeze': True})

    return sn_roi_act, sns, conn_trials


def get_module_cross_trialwise_z(conn_trials, p_mod0, p_mod1, trialwise=True,
                                 transpose=True):
    if transpose:
        conn_trials = np.transpose(conn_trials, (0, 1, 4, 2, 3))
    conn_trials_cross = get_partition_cross(conn_trials, p_mod0, p_mod1)
    flat_cross = np.reshape(conn_trials_cross, (conn_trials_cross.shape[0],
                                                conn_trials_cross.shape[1],
                                                conn_trials_cross.shape[2], -1))
    if trialwise:
        agg_zs = np.nanmean(flat_cross, axis=-1)
        agg_zs = np.nanmean(agg_zs, axis=1) # omit inc axis
        return agg_zs
    else:
        rs = np.nanmean(flat_cross, axis=2)
        agg_rs = np.nanmean(rs, axis=-1)
        return agg_rs


@timing
def get_dfs_conn_trials(fp='obj7_fMRI', single=False, squeeze=False,
                        combine_regions=False):


    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions
              }
    sn_inc_conn, sn_conn, age2idxs, sn_roi_act, df_sns_l = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache',
                    )
    sn_inc_activity_std = stdize(sn_roi_act, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    # sns = age2idxs[1] + age2idxs[2]
    sns = [df['sn'].iloc[0] for df in df_sns_l]

    if squeeze:
        sn_roi_act = np.nanmean(sn_roi_act, axis=1)
        # conn_trials = np.nanmean(conn_trials, axis=1)
        # return sn_roi_act, conn_trials, sns

    if single:
        return conn_trials[[0]], [df_sns_l[0]]
    else:
        return sn_roi_act, conn_trials, df_sns_l, sns


def get_hemi_vendor_df(fp='obj7_fMRI', scrub=False, anat=False,
                       anat_version=1):

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=anat, scrub=scrub,
                              anat_version=anat_version)

    atlas = get_atlas()
    ps = {'da': p_d_ant, 'dp': p_d_pos, 'va': p_v_ant, 'vp': p_v_pos,}
    ps_hemi = defaultdict(list)
    for key, p in ps.items():
        for i in p:
            coord = atlas['coords'][i]
            if coord[0] < 0:
                ps_hemi[f'L{key}'].append(i)
            else:
                ps_hemi[f'R{key}'].append(i)
    ps_hemi.update(ps)

    # if fp == 'rs_medium':
    #
    # else:

    # if fp == 'rs_medium':
    #     sn_roi_act, sns, conn_trials = load_a(fp=fp)
    #     conn_trials = conn_trials[:, None]
    #     # print(f'{conn_trials.shape=}')
    #     # quit()
    # else:
    _, conn_trials, df_sns_l, _ = get_dfs_conn_trials(fp)
        # print(f'{conn_trials.shape=}')
        # quit()

    new_cols = []

    for i, p0 in enumerate(ps_hemi):
        for j, p1 in enumerate(ps_hemi):
            # if i >= j:
            #     continue
            sn_agg_trials_dd = get_module_cross_trialwise_z(conn_trials,
                                                            ps_hemi[p0],
                                                            ps_hemi[p1])
            # if p0 == 'dp' and p1 == 'da':
                # print(sn_agg_trials_dd[:, 5])
                # print(sn_agg_trials_dd)
                # quit()
            for k, df_sn in enumerate(df_sns_l):
                df_sn[f'{p0}_{p1}'] = sn_agg_trials_dd[k, :]
                df_sn[f'{p0}_{p1}'] = stats.zscore(df_sn[f'{p0}_{p1}'],
                                                   nan_policy='omit')
            new_cols.append(f'{p0}_{p1}')
    df_sns = pd.concat(df_sns_l)
    return df_sns, new_cols


def get_sn_rs(sn, clean=True, compcor=True, light=False, medium=False,
              trad=False, near_OG=False, true_OG=False):
    fp_in = fr'fMRI_in/{sn}/resting/rs.nii.gz'
    img = image.load_img(fp_in)
    if true_OG:
        dir_mask = fr'E:\PycharmProjects_E\SchemeRep\fMRI_in\masks'
        fn_mask = fr'sub-{sn}_space-MNI152NLin2009cAsym_res-2_GrayMatter20.nii'
        fp_mask = fr'{dir_mask}\{fn_mask}'

        fp_in = fr'E:\PycharmProjects_E\SchemeRep\cache\confounds\{sn}_resting_confounds.tsv'
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
        dir_mask = fr'E:\PycharmProjects_E\SchemeRep\fMRI_in\masks'
        fn_mask = fr'sub-{sn}_space-MNI152NLin2009cAsym_res-2_GrayMatter20.nii'
        fp_mask = fr'{dir_mask}\{fn_mask}'

        fp_in = fr'E:\PycharmProjects_E\SchemeRep\cache\confounds\{sn}_resting_confounds.tsv'
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
        img = image.clean_img(img, confounds=df_confounds, high_pass=1/128,
                              standardize=False, t_r=2,
                              mask_img=fp_mask)
        # print('ook')
        # quit()
    data = img.get_fdata()
    return data


def get_sn_raw_enc(sn):
    fp_in = fr'G:\SchemeRep_raw_data_dir_preproc\{sn}\ENC\BOLD_run1.nii'
    try:
        img = image.load_img(fp_in)
    except ValueError:
        return np.full((97, 115, 97, 276), np.nan)
    confounds = pd.DataFrame(high_variance_confounds(img, percentile=1))
    img = image.clean_img(img, confounds=confounds)
    data = img.get_fdata()
    return data


def get_LSS_SchemeRep(sn, run=1, lsa=False, easy_override=False):
    if sn in ['116', '125', '135', '138']:
        data = np.full((97, 115, 97, 38), np.nan)
    else:
        data = get_LSS_img(sn, run, hcp=False, lsa=lsa, easy_override=easy_override)
    data_mask = sanity_load(sn)

    data_bads = np.isnan(data_mask)[..., :data.shape[-1]]
    num_nans = np.sum(data_bads[..., 0])
    print(f'{sn}, {num_nans=}')
    data[data_bads] = np.nan

    # print(data_mask.shape)
    # print(data.shape)
    # # quit()

    return data


def sanity_load(sn):
    df_sn = get_trial_info(sn)
    # print(df_sn[['obj_trial', 'obj']])
    # quit()
    data, _ = load_ni_w_nan_fps(df_sn['obj7_fMRI'])
    # data = img.get_fdata()
    return data


def load_resting_data(raw_enc=False, lss_enc=False, lsa=False, YA_only=False,
                      sanity=False, combine_regions=False, sns_key='loose',
                      compcor=True, clean=True, light=False, medium=False,
                      trad=False, near_OG=False, true_OG=False):

    age2sn = get_sns(sns_key)
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
    for i, sn in tqdm(enumerate(sns), desc='Loading fMRI'):
        logging.debug(f'{sn=}')
        if sanity:
            data = pickle_wrap(sanity_load, kwargs={'sn': sn})
        elif lss_enc:
            data = get_LSS_SchemeRep(sn, lsa=lsa, easy_override=False)
        elif raw_enc:
            data = pickle_wrap(get_sn_raw_enc, kwargs={'sn': sn})
        else:
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
        # quit()
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
    sn_roi_act, sns = pickle_wrap(f, easy_override=easy_override,
                                  kwargs={'YA_only': YA_only},
                                  RAM_cache=RAM_cache)
    # sn_roi_act = sn_roi_act[:, :, 4:] # bad trials to start?
    bad_rs_sns = {'133'}
    sns = [sn for sn in sns if sn not in bad_rs_sns]
    sn_roi_act = stdize(sn_roi_act, axis=2, nans=True)
    assert len(sn_roi_act.shape) == 3, f'More than 3 dims: {sn_roi_act.shape=}'
    if norm_std: sn_roi_act = normalize_std_over_time(sn_roi_act)
    conn_trials = sn_roi_act[..., None, :] * \
                  sn_roi_act[..., None, :, :]
    return sn_roi_act, sns, conn_trials
