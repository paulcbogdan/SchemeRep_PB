from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as stats

from Study1A.modularity_funcs import get_FC_between_ROIs
from atlas_utils import get_atlas
from Study1A.load_data_Study1A import load_FC
from utils import pickle_wrap, timing, stdize
from Study1A.partition_VD_PA import get_VD_PA_partitions


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
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache',
                    )
    sn_inc_activity_std = stdize(sn_roi_act, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    sns = [df['sn'].iloc[0] for df in df_sns_l]

    if squeeze:
        sn_roi_act = np.nanmean(sn_roi_act, axis=1)

    if single:
        return conn_trials[[0]], [df_sns_l[0]]
    else:
        return sn_roi_act, conn_trials, df_sns_l, sns


def get_hemi_vendor_df(fp='obj7_fMRI', scrub=False, anat=False,
                       anat_version=1):

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', do_PA=True, anat=anat,
                             anat_ver=anat_version)

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

    _, conn_trials, df_sns_l, _ = get_dfs_conn_trials(fp)
    new_cols = []

    for i, p0 in enumerate(ps_hemi):
        for j, p1 in enumerate(ps_hemi):
            sn_agg_trials_dd = get_FC_between_ROIs(conn_trials,
                                                   ps_hemi[p0],
                                                   ps_hemi[p1])
            for k, df_sn in enumerate(df_sns_l):
                df_sn[f'{p0}_{p1}'] = sn_agg_trials_dd[k, :]
                df_sn[f'{p0}_{p1}'] = stats.zscore(df_sn[f'{p0}_{p1}'],
                                                   nan_policy='omit')
            new_cols.append(f'{p0}_{p1}')
    df_sns = pd.concat(df_sns_l)
    return df_sns, new_cols
