# see fluctuations.py as of Monday June 24, should produce r = -.18 for dv_dv vs. dd_dd

from analyze_rs import prep_conn_ps
from atlas_utils import get_atlas
from load_more import load_a, get_module_cross_trialwise_z, get_dfs_conn_trials, load_resting_data, load_act_conn
from old.plot_gen import plot_connectivity
from old_Apr6.fluctuations import get_df_networks, partial_corr_df
from utils import pickle_wrap, stdize
from collections import defaultdict

from old_Apr6.vendor_lmers import get_module_trialwise_z
from scipy import stats
import pandas as pd
import numpy as np
import pingouin as pg

import matplotlib.pyplot as plt
from vendor_partitioning import get_vendor_partitions
from functools import partial
import os
os.chdir(r'H:\PycharmProjects_H\SchemeRep')


def produce_Fig5A(fp='rs_medium', anat_ver=2, combine_regions=False):
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
    # print(conn)
    conn = np.array(conn)
    conn[np.diag_indices_from(conn)] = np.nan
    ticks = list(np.arange(len(networks)))
    tick_labels = networks
    tick_lows = np.arange(len(networks))
    title = 'Resting-state avg. network correlations'

    # plot_connectivity(conn, ticks, tick_labels, tick_lows, title=title, no_avg=True, cbar_label='Pearson\'s r',
    #                   vmin=-0.5, vmax=0.5)


    networks = ['dd', 'vv', 'dv_ant', 'dv_pos',
                'dd_vv', 'dv_dv']

    networks += ['no_no']
    # networks += ['pd_no', 'ad_no', 'av_no', 'pv_no']
    print(df[networks].corr())

    rs = []
    for sn, df_sn in df.groupby('sn'):
        # key0 = 'dd_vv'
        # key1 = 'vv'
        # df_sn.dropna(subset=[key0, key1], inplace=True)
        # vals = df_sn[key0].sample(frac=1, ignore_index=True)
        # df_sn[key0] = vals.tolist()

        ar = partial_corr_df(df_sn, networks,
                             cov=['pd_no', 'ad_no', 'av_no', 'pv_no'])
        r = ar[4, 5]
        print(f'{r=:.3f}')
        # r = ar[0, 1]

        rs.append(r)
    M_rs = np.mean(rs)
    SE_rs = stats.sem(rs)
    M_low = M_rs - SE_rs * 1.96
    M_high = M_rs + SE_rs * 1.96
    print(f'{M_rs=:.3f} [{M_low:.3f}, {M_high:.3f}], {SE_rs=:.3f}')

    t, p = stats.ttest_1samp(rs, 0)
    print(f'One sample: {t=:.3f}, {p=:.10f}')

    quit()

    # 'dd', 'vv', 'dv_ant', 'dv_pos'

    from pymer4.models import Lmer
    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    model = Lmer('dd_vv ~ dv_dv + FC_all + (1 | sn)', data=df)
    model.fit(REML=True, verbose=False, summary=True)
    print(model.summary())

if __name__ == '__main__':
    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 1000)

    produce_Fig5A()
    # get_dfs_conn_trials(fp='obj7_fMRI', single=False)
    # analyze_networks()

