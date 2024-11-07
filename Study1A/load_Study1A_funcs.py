import random
from collections import defaultdict

import numpy as np
import scipy.stats as stats

import utils
from Utils.atlas_funcs import get_atlas, get_BNA_ROIs
from Utils.pickle_wrap_funcs import pickle_wrap
from org_sns import get_sns
from organize_bhv import get_trial_info


def load_FC(atlas_name='BNA', fp='obj7_fMRI', split=False, key='inc',
            key_vals=(1, 2, 3), odd_even=False,
            do_sort=False, voxelwise=False, regionwise=False,
            combine_regions=False, get_df_sn=False,
            combine_bilateral=False, strict_sns=False, ):
    if atlas_name == 'schaefer':
        atlas = get_atlas(schaefer=True)
    else:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=combine_bilateral,
                          split=split, split_code='xyz',
                          )

    # Some of this is related to using this code for other projects
    age2sn = get_sns('all' if strict_sns else fp, sh=False)

    for age, sns in age2sn.items():
        print(f'{age=}, {len(sns)=}')

    sn_inc_conn = []
    sn_inc_activity = []
    sn_conn = []
    age2idxs = defaultdict(list)
    sn_idx = 0
    ROI2act = defaultdict(list)
    Y = []
    grp_idxs = []
    df_sns = []
    for i, age in enumerate([1, 2]):
        sns = age2sn[age]
        if combine_regions:
            if combine_bilateral:
                ROIs_l = atlas['ROI_regions']
            else:
                ROIs_l = get_BNA_ROIs(code='BNA_region')
        else:
            if combine_bilateral:
                ROIs_l = atlas['ROIs']
            else:
                ROIs_l = get_BNA_ROIs(code=atlas_name)

        for sn in sns:
            print(f'Prepping FC: {sn=} (idx: {sn_idx})')
            df_sn = get_trial_info(sn, easy_override=False, ret=False)
            if do_sort:
                sess = fp.split('_')[0].replace('2', '').replace('3', ''). \
                    replace('4', '').replace('7', '')
                df_sn.sort_values(by=f'{sess}_trial', inplace=True)
            try:
                ROI2vecs0 = get_ROI_vecs(sn, atlas, fp, df_sn,
                                         nan_thresh=1.01,
                                         drop_nan_voxels=False,
                                         org_by_region=regionwise,
                                         easy_override=False,
                                         combine_regions=combine_regions)
            except TypeError as e:
                print(f'Error missing files ({sn}):', e)
                continue
            if voxelwise or regionwise:
                for ROI in ROI2vecs0:
                    ROI2act[ROI].append(ROI2vecs0[ROI])
                y = []
                for v in df_sn[key]:
                    for i, key_val in enumerate(key_vals):
                        if v == key_val:
                            y.append(i)
                            break
                    else:
                        y.append(np.nan)
                Y.extend(y)
                grp_idxs.extend([sn_idx] * 114)
                age2idxs[age].append(sn_idx)
                sn_idx += 1
                continue

            activity_ar = []
            for ROI in ROIs_l:
                if ROI not in ROI2vecs0:
                    activity_ar.append(np.full(114, np.nan))
                else:
                    activity_ar.append(np.nanmean(ROI2vecs0[ROI], axis=1))
            activity_ar = np.array(activity_ar)
            conn_no_cond = np.corrcoef(activity_ar)
            sn_conn.append(conn_no_cond)
            activity_inc = []
            conns = []

            nan_trials = np.any(np.isnan(activity_ar), axis=0)
            if sum(nan_trials) > 0:
                print('Participant has NaNs!')

            for i, inc in enumerate(key_vals):
                if key == 'rand':
                    if key_vals == (1, 2, 3):
                        matching_trials = df_sn['inc'].sample(frac=1.) == inc
                    else:
                        matching_trials = df_sn['vis_hit'].sample(frac=1.) == inc
                else:
                    matching_trials = df_sn[key] == inc
                if odd_even:
                    matching_trials_even = []
                    matching_trials_odd = []
                    cnt = 0
                    org = [False] * (sum(matching_trials) // 2) + \
                          [True] * (sum(matching_trials) // 2)
                    if sum(matching_trials) % 2:
                        org += [False]
                    random.shuffle(org)
                    for trial in matching_trials:
                        if trial:
                            matching_trials_even.append(org[cnt])
                            matching_trials_odd.append(not org[cnt])
                            cnt += 1
                        else:
                            matching_trials_even.append(False)
                            matching_trials_odd.append(False)
                    activity_inc.append([activity_ar[:, matching_trials_even],
                                         activity_ar[:, matching_trials_odd]])

                    conns.append([np.corrcoef(activity_ar[:, matching_trials_even]),
                                  np.corrcoef(activity_ar[:, matching_trials_odd])])
                else:
                    activity_ar_matched = np.full(activity_ar.shape, np.nan)
                    activity_ar_matched[:, matching_trials] = \
                        activity_ar[:, matching_trials]
                    activity_inc.append(activity_ar_matched)

                    # activity_ar_std = stdize(activity_ar_matched, axis=1,
                    #                          nans=True)
                    activity_ar_std = stats.zscore(activity_ar_matched, axis=1,
                                                   nan_policy='omit')
                    trial_z = (activity_ar_std[None, :, :] *
                               activity_ar_std[:, None, :])
                    conn_inc = np.nanmean(trial_z, axis=-1)
                    conns.append(conn_inc)

            conns = np.array(conns)
            sn_inc_conn.append(conns)
            sn_inc_activity.append(activity_inc)
            age2idxs[age].append(sn_idx)
            df_sns.append(df_sn)
            sn_idx += 1

    if voxelwise or regionwise:
        for ROI, vals in ROI2act.items():
            ROI2act[ROI] = np.array(vals)
            print(f'{ROI}, {ROI2act[ROI].shape=}')
        return ROI2act, Y, grp_idxs, age2idxs
    else:
        sn_inc_conn = np.array(sn_inc_conn)
        sn_conn = np.array(sn_conn)
        diag = np.diag_indices(sn_inc_conn.shape[-1])
        sn_inc_conn[..., diag[0], diag[1]] = np.nan
        sn_conn[..., diag[0], diag[1]] = np.nan
        sn_inc_activity = np.array(sn_inc_activity)
        if get_df_sn:
            return sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns
        else:
            return sn_inc_conn, sn_conn, age2idxs, sn_inc_activity


def get_ROI_vecs(sn, atlas, fp_fMRI_col, df_sn, nan_thresh=.25,
                 org_by_region=False, inc=None, drop_nan_voxels=True,
                 easy_override=False, combine_regions=False,
                 verbose=0):
    ROIs = atlas['ROIs']
    ROI_regions = atlas['ROI_regions']
    shenyang_key = atlas['shenyang']
    # shenyang_key = False
    n_ROIs = len(ROIs)
    n_regions = len(np.unique(ROI_regions))
    nan_str = f'_nan{nan_thresh}' if nan_thresh != .25 else ''
    nan_str += '_dropNaNvox' if drop_nan_voxels else ''
    sh_str = f'_sh' if shenyang_key else ''
    org_by_region_str = '_oByR' if org_by_region else ''
    inc_str = '' if inc is None else \
        '_Con' if inc == 1 else \
            '_Inc' if inc == 2 else '_Neu'
    combine_str = '_comb' if combine_regions else '_noComb'
    stim_order = ''.join(df_sn['obj'].values).replace(' ', '')
    stim_order = stim_order[::len(stim_order) // 10]
    fp_cache = fr'cache\ROI2vecs\sn{sn}_{fp_fMRI_col}{inc_str}_nROI{n_ROIs}' \
               fr'_reg{n_regions}{org_by_region_str}{nan_str}{combine_str}' \
               fr'{sh_str}{stim_order}.pkl'
    if verbose >= 0: print(f'Load ROI2vecs: {fp_cache=}')
    f = lambda: get_ROI_vecs_(df_sn, fp_fMRI_col, atlas, nan_thresh=nan_thresh,
                              org_by_region=org_by_region, drop_nan_voxels=drop_nan_voxels)
    r2vecs = pickle_wrap(f, fp_cache, easy_override=easy_override,
                         verbose=verbose)
    return r2vecs


def get_ROI_vecs_(df_sn, fp_fMRI_col, atlas,
                  nan_thresh=.25, org_by_region=False,
                  drop_nan_voxels=True):
    try:
        img, good_idxs = utils.load_ni_w_nan_fps(df_sn[fp_fMRI_col])
    except TypeError as e:
        sn = df_sn['sn'].values[0]
        print(df_sn[fp_fMRI_col])
        for i in range(10):
            print(f'Error in loading ({fp_fMRI_col}): {e}')
            print(f'\t{sn=}')
        raise TypeError

    n_nans = np.isnan(img).sum()
    print(f'Total number of NaNs: {n_nans / 114:.1f}')
    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    ROI2vecs = {}
    region2vecs = defaultdict(list)
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        region_vecs = img[atlas_roi]

        voxels_w_nan = np.isnan(region_vecs[:, good_idxs]).any(axis=1)

        n_nans_ROI = np.sum(voxels_w_nan)
        p_nan_any = n_nans_ROI / len(voxels_w_nan)
        n_nan_overall = np.sum(np.isnan(region_vecs[:, good_idxs]))
        p_nan_overall = n_nan_overall / np.prod(region_vecs[:, good_idxs].shape)

        if p_nan_any > nan_thresh:  # more than 10%
            print(f'Skip ({region}): {p_nan_any=:.2f}, {p_nan_overall=:.2f}')
            continue
        if drop_nan_voxels:
            region_vecs = region_vecs[~voxels_w_nan, :]
        region_vecs = region_vecs.T
        ROI2vecs[ROI] = region_vecs
        if org_by_region:
            region2vecs[region].append(np.nanmean(region_vecs, axis=1))

    region2vecs = dict(region2vecs)

    for region, l in region2vecs.items():
        region2vecs[region] = np.array(l).T

    if org_by_region:
        return region2vecs
    else:
        return ROI2vecs
