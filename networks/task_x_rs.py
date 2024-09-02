import numpy as np

from load_more import get_module_cross_trialwise_z
from old.network_funcs import load_FC_for_Lifu
from old_Apr6.fluctuations import partial_corr_df, get_df_networks
from utils import pickle_wrap, stdize
import pandas as pd

from vendor_partitioning import get_vendor_partitions
from scipy import stats
import matplotlib.pyplot as plt

def get_sn2r(fp='rs_medium', anat_ver=3, combine_regions=False):
    df, networks = pickle_wrap(get_df_networks,
                               kwargs={'fp': fp,
                                       'norm_std': False,
                                       'zscore': True,
                                       'anat_ver': anat_ver,
                                       'combine_regions': combine_regions},
                               easy_override=False)

    df['da_dp'] = df['da'] + df['dp']
    df['va_vp'] = df['va'] + df['vp']

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    networks = networks[:-4]
    conn = df[networks].corr()
    conn = np.array(conn)
    conn[np.diag_indices_from(conn)] = np.nan
    networks = ['dd', 'vv', 'dv_ant', 'dv_pos',
                'dd_vv', 'dv_dv']

    networks += ['no_no']
    print(df[networks].corr())

    rs = []
    sn2r = {}
    for sn, df_sn in df.groupby('sn'):
        ar = partial_corr_df(df_sn, networks,
                             cov=['pd_no', 'ad_no', 'av_no', 'pv_no',
                                  ], verbose=0)
        r = ar[4, 5]
        sn2r[sn] = r
        print(f'{sn} | {r=:.2f}')
    return sn2r

def get_sn2ef(fp='obj7_fMRI', anat=True, weighted=False,
              anat_ver=3):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', anat=anat, weighted=weighted,
                              flip=True, thr=.9, scrub=False, anat_ver=anat_ver)
    n_rois = sn_inc_activity.shape[2]
    # matrix_mask = np.ones((n_rois, n_rois), dtype=bool)
    matrix_mask[~matrix_mask] = np.nan

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)
    conn_trials = sn_inc_activity_std[..., None, :] * \
                  sn_inc_activity_std[..., None, :, :]
    conn_trials = np.repeat(matrix_mask[None, None, ..., None],
                            conn_trials.shape[-1], axis=4) * conn_trials[..., :]

    dd_flat = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_d_ant,
                                             trialwise=False)
    vv_flat = get_module_cross_trialwise_z(conn_trials, p_v_pos, p_v_ant,
                                             trialwise=False)
    dv_ant = get_module_cross_trialwise_z(conn_trials, p_d_ant, p_v_ant,
                                            trialwise=False)
    dv_pos = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_v_pos,
                                            trialwise=False)

    dv_cross = get_module_cross_trialwise_z(conn_trials, p_d_pos, p_v_ant,
                                            trialwise=False)
    vd_cross = get_module_cross_trialwise_z(conn_trials, p_v_pos, p_d_ant,
                                            trialwise=False)

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

    # keys = ['Within', 'Between', 'dd', 'vv', 'dv_ant', 'dv_pos',
    #         'dv_cross', 'vd_cross']
    keys = ['dd_vv', 'dv_dv', 'dd', 'vv', 'dv_ant', 'dv_pos',
            'dv_cross', 'vd_cross']
    data = [agg_within, agg_between, dd_flat, vv_flat, dv_ant, dv_pos,
            dv_cross, vd_cross]
    for key, flat in zip(keys, data):
        incs += ['Inc'] * n_sn + ['Neu'] * n_sn + ['Con'] * n_sn
        ages += (['YA']*len(age2idxs[1]) + ['OA']*len(age2idxs[2]))*3
        wbs += [key] * (3 * n_sn)
        subj_nums += list(range(n_sn)) * 3
        vals += list(flat.T.reshape(-1))

    print(f'{len(vals)=}, {len(incs)=}, {len(ages)=}, {len(wbs)=}')

    d = {'vals': vals, 'inc': incs, 'within_between': wbs,
         'age': ages, 'sn': subj_nums}

    df_agg = pd.DataFrame(d)

    df = df_agg.groupby(['inc', 'within_between', 'sn'])[['vals']].mean()
    df['cnt'] = df_agg.groupby(['inc', 'within_between', 'sn'])['vals'].count()
    assert (df['cnt'] < 2).all()

    sns_idxs = df_agg['sn'].unique()
    intr = (df.loc[('Inc', 'dv_dv', sns_idxs)].values -
            df.loc[('Con', 'dv_dv', sns_idxs)].values -
            df.loc[('Inc', 'dd_vv', sns_idxs)].values +
            df.loc[('Con', 'dd_vv', sns_idxs)].values)
    intr = intr[:, 0]

    sns = [df_sn['sn'].iloc[0] for df_sn in df_sns]

    assert len(sns_idxs) == len(sns)

    sn2ef = {sn: ef for sn, ef in zip(sns, intr)}
    return sn2ef


def corr_task_x_rs():
    sn2r = get_sn2r()
    sn2ef = get_sn2ef()

    sns = set(sn2ef.keys()).intersection(sn2r.keys())
    rs = [sn2r[sn] for sn in sns]
    efs = [sn2ef[sn] for sn in sns]


    # print(f'{rs=}')
    # print(f'{efs=}')

    sns_ = []
    rs_ = []
    efs_ = []
    for sn, r, ef in zip(sns, rs, efs):
        if ef < .3:
            sns_.append(sn)
            rs_.append(r)
            efs_.append(ef)
        print(f'{sn=}, {r=:.2f}, {ef=:.2f}')
    sns = sns_
    rs = rs_
    efs = efs_

    r, p = stats.spearmanr(rs, efs)
    print(f'{r=:.3f}, {p=:.3f}')

    plt.scatter(rs, efs)
    plt.title(f'Intensity of RS fluctuation x PE effect in task: '
              f'{r=:.2f}')
    plt.ylabel('Task interactin effect\n'
               '(High PE [VD] - Low PE [VD] -\nHigh PE [PA] + Low PE [PA])')
    plt.xlabel('rs-fluc correlation')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    corr_task_x_rs()
    # sn2ef = get_sn2ef()
    # import sys
    # sys.path.append('..')











