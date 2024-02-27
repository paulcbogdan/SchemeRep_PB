from analyze_rs import get_rs_vendor_df
from atlas_utils import get_atlas

from functools import cache
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fluctuations import get_df_networks
from old.modularity import get_partition_cross, get_partition_matrix
from old.network_funcs import load_FC_for_Lifu
from vendor_lmers import get_vendor_df
from vendor_partitioning import get_vendor_partitions
from utils import timing, pickle_wrap, stdize, get_formula_cols
import scipy.stats as stats
from warnings import filterwarnings
import os

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

def get_all_task_vendor(zscore=True, anat=False):
    fps = ['bl7_fMRI', 'con7_fMRI', 'rs', 'vis7_fMRI', 'obj7_fMRI']
    df_l = []
    for fp in fps:
        if fp == 'rs':
            if anat:
                # df, networks = pickle_wrap(get_df_networks,
                #                            kwargs={'fp': 'rs',
                #                                    'norm_std': False,
                #                                    'zscore': zscore},
                #                            easy_override=False)
                df, vndr_cols = pickle_wrap(get_rs_vendor_df,
                                            kwargs={'roiwise': False,
                                                    'do_hemi': False,
                                                    'zscore': True,
                                            'high_var_confounds': False},
                                            easy_override=False)
                df['task'] = 'RS'
        else:
            df, vndr_cols = pickle_wrap(get_vendor_df, None,
                                        kwargs={'fp': fp, 'scrub': False,
                                                'zscore': zscore, 'anat': anat},
                                        easy_override=False)
            df['task'] = fp.split('_')[0][:-1].upper()
        df_l.append(df)
    df = pd.concat(df_l).reset_index(drop=True)
    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']


    # df['dd_vv'] = stats.zscore(df['dd_vv'])
    # df['dv_dv'] = stats.zscore(df['dv_dv'])
    # df['vv'] = stats.zscore(df['vv'])
    return df


def vendor_lmer_BL_resp(fp='bl7_fMRI'):
    # df, vndr_cols = pickle_wrap(get_vendor_df, None,
    #                             kwargs={'fp': fp, 'scrub': False,
    #                                     'zscore': True,
    #                                     'anat': True},
    #                             easy_override=False, cache_dir='cache')
    # df['bl_resp'] = df['bl_resp'].apply(
    #     lambda x: x[0] if isinstance(x, np.ndarray) else x)
    # df['task'] = 'BL'
    #
    # df_rest, networks = pickle_wrap(get_df_networks, kwargs={'fp': 'rs',
    #                                                     'norm_std': False,
    #                                                     'zscore': True},
    #                            easy_override=False)
    # df_rest['task'] = 'RS'
    # df = pd.concat([df, df_rest]).reset_index(drop=True)



    df = get_all_task_vendor(zscore=True, )

    # df = df[df['task'] == 'RS']

    # cols = ['dd', 'dv_pos']
    # for col in cols:
    #     df[f'{col}_z'] = stats.zscore(df[col], nan_policy='omit')

    # df = df[df['dd_z'].abs() < 5]
    # df = df[df['dv_pos_z'].abs() < 5]

    # plt.scatter(df['dv_pos'], df['dd'])
    # plt.show()

    #  FC_all * task +

    # df = df[df['task'] == 'RS']

    df['dp_va'] = df['dp'] * df['va']
    df['dp_va'] = stats.zscore(df['dp_va'], nan_policy='omit')
    df['da_vp'] = df['da'] * df['vp']
    df['da_vp'] = stats.zscore(df['da_vp'], nan_policy='omit')

    df['vv'] = stats.zscore(df['vv'], nan_policy='omit')
    df['dd'] = stats.zscore(df['dd'], nan_policy='omit')

    formula = ('dv_pos ~ vv * task + '
               '(1 | sn)')

    # However, this is only the case for RS
    # formula = ('dv_pos ~ vv * dp_va + '
    #            '(1 + vv * dp_va | sn)')

    formula = ('dv_pos ~ dd * da_vp + '
               '(1 + dd * da_vp | sn)') # sum of dd + da_vp = dd:da_vp coef

    # formula = ('dp ~ va * task + '
    #            '(1 | sn)')

    # formula = ('dv_pos ~ dd + FC_all + '
    #            '(1 + dd + FC_all | sn)')

    # formula = ('dd_vv ~ dv_dv * task +'
    #            '(1 | sn)')

    from pymer4 import Lmer
    cols = get_formula_cols(df, formula)
    df_vals = df[cols].dropna()
    for col in cols:
        if 'sn' in col or 'task' in col: continue
        df_vals[col] = stats.zscore(df_vals[col])
    model = Lmer(formula, data=df_vals)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())


if __name__ == '__main__':
    vendor_lmer_BL_resp()