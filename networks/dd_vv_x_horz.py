from dFC_control_FC import get_all_task_vendor
from factor_analysis import identify_extremely_low_variance_sn

import pandas as pd
from tqdm import tqdm

from analyze_rs import get_rs_vendor_df
from autocorr import add_prev, add_next
from dFC_control_FC import get_all_task_vendor
from utils import pickle_wrap
from vendor_lmers import get_vendor_df
from sklearn import decomposition
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

def do_dd_vv_x_horz(fp='rs'):
    # df = load_FA(fp)
    df = get_all_task_vendor(anat=True, zscore=True)
    # df = df[df['task'] != 'RS']
    # df = df[df['task'] == 'RS']
    bad_sns = identify_extremely_low_variance_sn(df)
    bad_sns.add('138')
    bad_sns.add('217') # almost always near zero for Horz
    bad_sns.add('110') # almost always nere zero for Vendor
    df = df[~df['sn'].isin(bad_sns)]
    df.dropna(subset=['da', 'dp', 'va', 'vp'], inplace=True)

    # df['dd_vv'] = df['dd'] + df['vv']
    # df['dv_dv'] = df['dv_ant'] + df['dv_pos']

    df['vert'] = df['da'] + df['dp'] - df['va'] - df['vp']
    df['horz'] = df['da'] - df['dp'] + df['va'] - df['vp']
    df['diag_a'] = df['da'] - df['dp'] - df['va'] + df['vp']
    df['diag_p'] = -df['da'] + df['dp'] + df['va'] - df['vp']

    df['up'] = df['da'] + df['dp']
    df['down'] = df['va'] + df['vp']
    df['ant'] = df['da'] + df['va']
    df['pos'] = df['dp'] + df['vp']


    df['vert'] /= np.std(df['vert'])
    df['horz'] /= np.std(df['horz'])

    df['dd_vv'] = df['dd'] + df['vv']
    # df['dd_vv'] = stats.zscore(df['dd_vv'], nan_policy='omit')
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    # df['dv_dv'] = stats.zscore(df['dv_dv'], nan_policy='omit')
    df['vendor'] = df['dd_vv'] - df['dv_dv']
    df['alt'] = df['dd'] - df['vv'] + df['dv_ant'] - df['dv_pos']

    df = df[df['task'] == 'OBJ']
    df['abs_horz'] = df['horz'].abs()
    df['da2'] = df['da']
    df['dp2'] = df['dp']
    df['horz'] = stats.zscore(df['horz'], nan_policy='omit')
    formula = ('dp ~ 1 + da*horz + inc + '
               '(1 + da*horz + inc | sn)')
    from pymer4.models import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())


if __name__ == '__main__':
    do_dd_vv_x_horz()