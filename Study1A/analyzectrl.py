import numpy as np
from Study1A.plot_Fig2F_boxes import get_FC_between_ROIs


import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from Study1A.load_Study1A_funcs import load_FC
from Utils.pickle_wrap_funcs import pickle_wrap
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions

from warnings import filterwarnings
import statsmodels.formula.api as smf


filterwarnings('ignore', category=SettingWithCopyWarning, )

np.float = float
np.bool = bool
np.int = int

def do_analyze_ctrl(schaefer=False):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': ('schaefer', 400) if schaefer else 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    if schaefer:
        schaefer = (schaefer, 400)
    else:
        schaefer = False
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True, schaefer=schaefer)
    matrix_mask[~matrix_mask] = np.nan

    sn_inc_activity_std = stats.zscore(sn_inc_activity, axis=3, nan_policy='omit')

    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    conn_trials = np.repeat(matrix_mask[None, None, ..., None],
                            conn_trials.shape[-1], axis=4) * conn_trials[..., :]

    dd_flat = get_FC_between_ROIs(conn_trials, p_d_pos, p_d_ant, trialwise=False)
    vv_flat = get_FC_between_ROIs(conn_trials, p_v_pos, p_v_ant, trialwise=False)
    dv_ant = get_FC_between_ROIs(conn_trials, p_d_ant, p_v_ant, trialwise=False)
    dv_pos = get_FC_between_ROIs(conn_trials, p_d_pos, p_v_pos, trialwise=False)
    dv_cross = get_FC_between_ROIs(conn_trials, p_d_pos, p_v_ant, trialwise=False)
    vd_cross = get_FC_between_ROIs(conn_trials, p_v_pos, p_d_ant, trialwise=False)
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': ('schaefer', 400) if schaefer else 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    if schaefer:
        schaefer = (schaefer, 400)
    else:
        schaefer = False
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True, schaefer=schaefer)
    matrix_mask[~matrix_mask] = np.nan

    sn_inc_activity_std = stats.zscore(sn_inc_activity, axis=3, nan_policy='omit')

    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    conn_trials = np.repeat(matrix_mask[None, None, ..., None],
                            conn_trials.shape[-1], axis=4) * conn_trials[..., :]

    dd_flat = get_FC_between_ROIs(conn_trials, p_d_pos, p_d_ant, trialwise=False)
    vv_flat = get_FC_between_ROIs(conn_trials, p_v_pos, p_v_ant, trialwise=False)
    dv_ant = get_FC_between_ROIs(conn_trials, p_d_ant, p_v_ant, trialwise=False)
    dv_pos = get_FC_between_ROIs(conn_trials, p_d_pos, p_v_pos, trialwise=False)
    dv_cross = get_FC_between_ROIs(conn_trials, p_d_pos, p_v_ant, trialwise=False)
    vd_cross = get_FC_between_ROIs(conn_trials, p_v_pos, p_d_ant, trialwise=False)

    flat_within = np.stack((dd_flat, vv_flat), axis=-1)
    agg_within = np.nanmean(flat_within, axis=-1)

    flat_between = np.stack((dv_ant, dv_pos), axis=-1)
    agg_between = np.nanmean(flat_between, axis=-1)

    n_sn = agg_within.shape[0]

    incs = []
    ages = []
    wbs = []
    subj_nums = []
    vals = []

    keys = ['Within', 'Between', 'dd', 'vv', 'dv_ant', 'dv_pos',
            'dv_cross', 'vd_cross']
    data = [agg_within, agg_between, dd_flat, vv_flat, dv_ant, dv_pos,
            dv_cross, vd_cross]
    print(agg_within.shape)
    # print(agg_within)
    # print(np.reshape(agg_within.T, -1))
    #
    # quit()

    # subj_nums = list(range(n_sn)) * 3
    # incs = ['Inc'] * n_sn + ['Neu'] * n_sn + ['Con'] * n_sn
    #
    # d = {'vals': vals, 'inc': incs, 'within_between': wbs,
    #      'age': ages, 'sn': subj_nums}
    #
    # for key, flat in zip(keys, data):
    #     wbs += [key] * (3 * n_sn)
    #     # vals += list(flat.T.reshape(-1))
    #     vals = flat.T.reshape(-1)

    keys = ['Within', 'Between', 'dd', 'vv', 'dv_ant', 'dv_pos',
            'dv_cross', 'vd_cross']
    data = [agg_within, agg_between, dd_flat, vv_flat, dv_ant, dv_pos,
            dv_cross, vd_cross]
    for key, flat in zip(keys, data):
        # incs += ['Inc'] * n_sn + ['Neu'] * n_sn + ['Con'] * n_sn
        incs += [-1] * n_sn + [0] * n_sn + [1] * n_sn
        ages += (['YA'] * len(age2idxs[1]) + ['OA'] * len(age2idxs[2])) * 3
        wbs += [key] * (3 * n_sn)
        subj_nums += list(range(n_sn)) * 3
        vals += list(flat.T.reshape(-1))

    d = {'vals': vals, 'inc': incs, 'within_between': wbs,
         'age': ages, 'sn': subj_nums}

    df_agg = pd.DataFrame(d)
    df_agg['sn'] = df_agg['sn'].astype(str)

    df_agg = df_agg[df_agg['within_between'].isin({'Within', 'Between'})]
    formula = 'vals ~ 1 + within_between * sn + within_between * inc'
    model = smf.ols(formula=formula, data=df_agg)
    res = model.fit()
    print(res.summary())


if __name__ == '__main__':
    do_analyze_ctrl()