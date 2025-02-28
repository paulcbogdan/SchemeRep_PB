import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np

from Utils.pickle_wrap_funcs import pickle_wrap
from Study1B.analyze_plot_Fig3 import get_wl_contrast_conn, make_conn, get_combo
from Study1B.BOLD_study1B import get_study1b_ar
from tqdm import tqdm
from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats


def get_ar_high_low(sn, kw):
    kw['only'] = 'loss'
    ar_high_lr_loss, ar_low_lr_loss = (
        get_study1b_ar(sn, rl_lr='lr', **kw))
    ar_high_rl_loss, ar_low_rl_loss = (
        get_study1b_ar(sn, rl_lr='rl', **kw))
    ar_high_loss = (np.nanmean(ar_high_lr_loss, axis=1) +
                    np.nanmean(ar_high_rl_loss, axis=1))
    ar_low_loss = (np.nanmean(ar_low_lr_loss, axis=1) +
                   np.nanmean(ar_low_rl_loss, axis=1))

    kw['only'] = 'win'
    ar_high_lr_win, ar_low_lr_win = (
        get_study1b_ar(sn, rl_lr='lr', **kw))
    ar_high_rl_win, ar_low_rl_win = (
        get_study1b_ar(sn, rl_lr='rl', **kw))
    ar_high_win = (np.nanmean(ar_high_lr_win, axis=1) +
                    np.nanmean(ar_high_rl_win, axis=1))
    ar_low_win = (np.nanmean(ar_low_lr_win, axis=1) +
                   np.nanmean(ar_low_rl_win, axis=1))
    ar_high = ar_high_loss + ar_high_win
    ar_low = ar_low_loss + ar_low_win
    return ar_high, ar_low

def get_conn_cols(conn, p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos):
    dd = conn[:, *np.ix_(p_d_pos, p_d_ant)]
    dd = np.nanmean(dd, axis=(1, 2))
    vv = conn[:, *np.ix_(p_v_pos, p_v_ant)]
    vv = np.nanmean(vv, axis=(1, 2))
    dv_ant = conn[:, *np.ix_(p_d_ant, p_v_ant)]
    dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    dv_pos = conn[:, *np.ix_(p_d_pos, p_v_pos)]
    dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    M_overall = np.nanmean(conn, axis=(1, 2))

    return dd, vv, dv_ant, dv_pos, M_overall

def do_analyze_ctrl_Study1B(schaefer=False, combine_regions=False):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True,
                             do_PA=True, thr=.9, anat_ver=3,
                             combine_regions=combine_regions,
                             schaefer=schaefer)

    kw = {'combine_regions': combine_regions, 'bilateral': False,
          'only': 'combo', 'learning_rate': 0.3,
          'drop_first': False, 'reset_trial0': True, }

    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')
    ar_high_l = []
    ar_low_l = []
    for sn in tqdm(sns):
        ar_high, ar_low = get_ar_high_low(sn, kw)
        ar_low_l.append(ar_low)
        ar_high_l.append(ar_high)
    ar_low = np.array(ar_low_l)
    ar_high = np.array(ar_high_l)
    pd_act = np.nanmean(ar_low[:, p_d_pos], axis=1)
    pv_act = np.nanmean(ar_low[:, p_v_pos], axis=1)
    ad_act = np.nanmean(ar_low[:, p_d_ant], axis=1)
    av_act = np.nanmean(ar_low[:, p_v_ant], axis=1)
    # print(pv_act.shape)
    # quit()

    if isinstance(schaefer, tuple):
        combine_regions = (combine_regions, ('schaefer', schaefer[1]))
    elif schaefer:
        combine_regions = (combine_regions, 'schaefer')
    kw = {'combine_regions': combine_regions, 'bilateral': False,
          'only': 'combo', 'num_sns': 1000, 'learning_rate': 0.3,
          'drop_first': False, 'reset_trial0': True, }

    if kw['only'] == 'wl':
        conn_highs, conn_lows, sns = (
            get_wl_contrast_conn(kw, easy_override=False))
    elif kw['only'] == 'combo':
        # can either be run while averaging a loss matrix & win matrix ('combo')
        #   or just making a single one covering both PE ('both')
        # the manuscript uses 'combo'
        conn_highs, conn_lows, sns = (
            get_combo(kw, easy_override=False))
    else:
        conn_highs, conn_lows, sns = (
            pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    if schaefer:
        combine_regions = combine_regions[0]
    print(conn_lows.shape)

    print('Grabbing conns...')
    dd_l, vv_l, dv_ant_l, dv_pos_l, M_overall_l = (
        get_conn_cols(conn_lows, p_dorsal, p_ventral, p_d_ant,
                      p_d_pos, p_v_ant, p_v_pos))
    dd_h, vv_h, dv_ant_h, dv_pos_h, M_overall_h = (
        get_conn_cols(conn_highs, p_dorsal, p_ventral, p_d_ant,
                      p_d_pos, p_v_ant, p_v_pos))
    PA_l = dd_l + vv_l
    VD_l = dv_ant_l + dv_pos_l
    PA_h = dd_h + vv_h
    VD_h = dv_ant_h + dv_pos_h

    vals = list(PA_l) + list(VD_l) + list(PA_h) + list(VD_h)
    incs = ['low'] * len(sns) * 2 + ['high'] * len(sns) * 2
    wbs = ['PA'] * len(sns) + ['VD'] * len(sns) + ['PA'] * len(sns) + ['VD'] * len(sns)
    pd_act = list(pd_act) * 4
    pv_act = list(pv_act) * 4
    ad_act = list(ad_act) * 4
    av_act = list(av_act) * 4
    sns = list(sns) * 4


    d = {'vals': vals, 'inc': incs, 'within_between': wbs, 'sn': sns,
         'pd': pd_act, 'pv': pv_act, 'ad': ad_act, 'av': av_act}

    for key, l in d.items():
        print(f'{key=}, {len(l)=}')

    df_agg = pd.DataFrame(d)
    df_agg['sn'] = df_agg['sn'].astype(str) # within_between * sn + within_between * inc +
    df_agg['vals'] = stats.zscore(df_agg['vals'])

    formula = 'vals ~ 1 + sn + within_between * inc + pd + pv + ad + av'
    model = smf.ols(formula=formula, data=df_agg)
    res = model.fit()
    print(res.summary())

    quit()


if __name__ == '__main__':
    do_analyze_ctrl_Study1B()
