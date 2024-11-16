import os
import pathlib

from Study1A.modularity_funcs import get_partition_matrix, get_modules

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np
import pandas as pd

from Study2A.rs_connectivity_funcs import get_df_networks, partial_corr_df
from Utils.plotting_funcs import plot_connectivity
from Utils.pickle_wrap_funcs import pickle_wrap

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def plot_Fig4B(anat_version=3):
    df, _ = pickle_wrap(get_df_networks, kwargs={'anat_ver': anat_version, },
                        easy_override=False)

    cols_order = ['Lda_Ldp', 'Rda_Rdp',
                  'Lva_Lvp', 'Rva_Rvp',
                  'Ldp_Lvp', 'Rdp_Rvp',
                  'Lda_Lva', 'Rda_Rva',
                  'Ldp_Rdp', 'Lvp_Rvp',
                  'Lda_Rda', 'Lva_Rva', ]

    df.dropna(subset=cols_order, inplace=True)

    for col in cols_order:
        M = df[col].mean()
        if col[1:3] != col[5:7]:
            continue

        height = '*' * int(abs(M) * 100)
        print(f'{col}: {M:+.2f} | {height}')

    tick_lows = np.arange(0, len(cols_order))
    ticks = tick_lows
    tick_labels = cols_order

    corr = partial_corr_df(df.copy(), cols_order,
                           cov=['pd_no_L', 'ad_no_L', 'av_no_L', 'pv_no_L',
                                'pd_no_R', 'ad_no_R', 'av_no_R', 'pv_no_R', ])

    corr[corr > .99] = np.nan

    fp = 'result_pics/Fig4/Fig4B_matrix.png'
    plot_connectivity(-corr, ticks, tick_labels, tick_lows,
                      title=None, no_avg=True, vmin=-0.15, vmax=0.15,
                      minimal=True, cmap='turbo_r', fp=fp)

    bool_ar = np.zeros(corr.shape)
    for i in range(corr.shape[0]):
        row = corr[i]
        median = np.nanmedian(row)
        bool_ar[i, row > median - .0001] += 1
        bool_ar[row > median - .0001, i] += 1
    bool_ar[bool_ar > 1.5] = 1
    corr = bool_ar

    partitions = get_modules(corr)
    corr_v0 = get_partition_matrix(np.ones(corr.shape), partitions[0],
                                   w_zeros=True)
    corr_v1 = get_partition_matrix(np.ones(corr.shape), partitions[1],
                                   w_zeros=True)

    plot_connectivity(corr_v0, ticks, tick_labels, tick_lows, title='',
                      no_avg=True, vmin=-0.3, vmax=0.3)

    plot_connectivity(corr_v1, ticks, tick_labels, tick_lows, title='',
                      no_avg=True, vmin=-0.3, vmax=0.3)


if __name__ == '__main__':
    pd.set_option('display.precision', 2)
    pd.options.display.float_format = '{:.2f}'.format

    plot_Fig4B()
