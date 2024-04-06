import os

from connRSA.conn_ISPC import run_settings_ISPC
from connRSA.conn_plot import plot_pie_chart
from corr_RSA_x_vendor import get_plain_df_sn

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

from collections import defaultdict

from conn_utils import get_BNA_ROIs
from organize_bhv import get_trial_info
from single_trial_conn import run_settings
from conn_report import report_results
from old.networks import prep_networks
from utils import pickle_wrap
import pandas as pd
import numpy as np
from tqdm import tqdm

import scipy.stats as stats
from colorama import Fore
import statsmodels.formula.api as smf
from pprint import pprint
import matplotlib.pyplot as plt


def pie_charts_ISPC(df, title):
    df = df[df['fp_idx'] == '0']

    df = df.groupby('img')[['conn_score', 'BOLD_score']].mean()

    df['conn_score'] /= df['conn_score'].std()
    df['BOLD_score'] /= df['BOLD_score'].std()

    mod = smf.ols(formula='conn_score ~ 1', data=df)
    res = mod.fit()
    conn_itr0 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'{conn_itr0=:.4f} ({p=:.3f}, {z=:.3f})')

    mod = smf.ols(formula='conn_score ~ 1 + BOLD_score', data=df)
    res = mod.fit()
    conn_itr_r = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress out BOLD: {conn_itr_r=:.4f} ({p=:.3f}, {z=:.3f})')
    print('-')

    mod = smf.ols(formula='BOLD_score ~ 1', data=df)
    res = mod.fit()
    bold_itr0 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'{bold_itr0=:.4f} ({p=:.3f}, {z=:.3f})')

    mod = smf.ols(formula='BOLD_score ~ 1 + conn_score', data=df)
    res = mod.fit()
    bold_itr_r = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress conn: {bold_itr_r=:.4f} ({p=:.3f}, {z=:.3f})')
    plot_pie_chart([conn_itr_r, bold_itr_r], title)
    return

    # plt.title(title, fontsize=24)
    # if conn_itr_r < 0:
    #     plt.pie([bold_itr_r], labels=['Region'],
    #             colors=[' red'],
    #             textprops=dict(color="w", fontsize=24, ha='center'),
    #             labeldistance=0.001,
    #             startangle=90,)
    # else:
    #     plt.pie([conn_itr_r, bold_itr_r],
    #                                        labels=['Conn', 'Region'],
    #             colors=['dodgerblue', 'red'], explode=[.01, .01],
    #             textprops=dict(color="w", fontsize=24, ha='center'),
    #             labeldistance=0.5,
    #             startangle=90,)
    # # plt.setp(autotexts, size=8, weight="bold")
    # plt.show()

def lmer_ISPC(df):
    for fp_idx in range(4):
        df_ = df[df['fp_idx'] == str(fp_idx)]
        df_['mem'] = stats.zscore(df_['mem'])
        mod = smf.ols(formula='BOLD_score ~ 1 + mem', data=df_)
        res = mod.fit()
        print(res.summary())
        # quit()
    quit()
    mod = smf.ols(formula='conn_score ~ 0 + mem * fp_idx', data=df)
    res = mod.fit()
    print(res.summary())

    # df = df.groupby('img')[['conn_score', 'BOLD_score']].mean()
    # mod = smf.ols(formula='conn_score ~ 1 + BOLD_score', data=df)
    # res = mod.fit()
    # print(res.summary())
    quit()

    formula = 'conn_score ~ 1 + BOLD_score + (1|fp_idx)'
    from pymer4.models import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=True)
    print(model.summary())

    # scores_conn = results_conn['scores_by_ROI']
    # print(f'{scores_conn.shape=}')
    # scores_conn = np.array(scores_conn)
    # # if not ISPC:
    # scores_sep = results_bold_sep['scores_by_ROI']
    # scores_sep = np.array(scores_sep)
    # scores_comb = results_bold_com['scores_by_ROI']


    # print(df)
    # print(results_conn['scores_by_ROI'].shape)
    quit()

    # scores_by_ROI = np.array(results['scores_by_ROI'])
    # scores_by_fp = np.transpose(scores_by_ROI, (0, 2, 1))

    # df, ROI_cols = organize_df(results_conn, results_bold_sep,
    #                            results_bold_comb, do_networks, target_ROI,
    #                            ISPC=True)
    #
    # lmer_stats(df, ROI_cols)

def load_for_lmer_ISPC(four_tasks='8', conn='euc', combine_regions=False,
			 split=False, do_networks=0, trial_similarity='corr',
			 age=1):
    settings = locals().copy()
    dir_results = r'cache/conn_RSA'
    results_conn = pickle_wrap(run_settings_ISPC, None, kwargs=settings,
                               easy_override=False, cache_dir=dir_results)
    report_results(results_conn, do_lmer=True, ISPC=True)

    if ('RDM_method' in settings and (settings['RDM_method'] is not None) and
            'complex_mean' in settings['RDM_method']):
        settings['RDM_method'] = 'within_nan'
    #
    settings['conn'] = 'BOLD'
    results_bold_comb = pickle_wrap(run_settings_ISPC, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results)

    report_results(results_bold_comb, do_lmer=True, ISPC=True)

    results_bold_sep = {}

    # settings['do_networks'] = False
    # results_bold_sep = pickle_wrap(run_settings_ISPC, None,
    #                                kwargs=settings, easy_override=False,
    #                                verbose=1, cache_dir=dir_results)

    # report_results(results_bold_sep, do_lmer=False)


    return results_conn, results_bold_comb, results_bold_sep


def run_lmer_PFC_ISPC():
    # target_ROI = 'else'

    df_bhv, _ = get_plain_df_sn()
    df_bhv = df_bhv.groupby('obj')['hit_hit'].mean()
    mem_scores = list(df_bhv.values) * 4

    target_ROI = 'MTL'

    ROI2network = {'Occipital': (1, 0), 'Ventral': (1, 1), 'Dorsal': (1, 1),
                   'else_cortical': (2, 1),
                   'PFC': (16, 0),
                   'PFC_ACC': (14, 1), 'FP': (14, 2),
                   'perceptual': 17,
                   'full_frontal': 11, 'full_frontal_CG': 11,
                   'MTL': (9, 0)}

    do_networks = ROI2network[target_ROI][0]
    results_conn, results_bold_comb, results_bold_sep = (
        load_for_lmer_ISPC(four_tasks='8', conn='prod', combine_regions=False,
                           split=False, do_networks=do_networks,
                           trial_similarity='corr',
                           age='healthy'))

    idx = ROI2network[target_ROI][1]
    results_conn['scores_by_ROI'] = results_conn['scores_by_ROI'][:, idx, :]
    results_bold_comb['scores_by_ROI'] \
        = results_bold_comb['scores_by_ROI'][:, idx, :]

    # networks = {'PFC': ['SFG', 'MFG', 'IFG', 'OrG'],
    #             'PFC_ACC': ['SFG', 'MFG', 'IFG', 'OrG', 'ACC'],
    #             'FP': ['MFG', 'IFG', 'IPL', 'SPL']}



    d = {'conn_score': np.reshape(results_conn['scores_by_ROI'], -1),
         'BOLD_score': np.reshape(results_bold_comb['scores_by_ROI'], -1),
         'fp_idx': list(range(4)) * 114,
         'img': list(range(114)) * 4,
         'mem': mem_scores}

    df = pd.DataFrame(d)
    df['fp_idx'] = df['fp_idx'].astype(str)

    title = f'ISPC: {target_ROI}'
    title = title.replace('PFC_ACC', 'Prefrontal')
    title = title.replace('MTL', 'Medial Temporal Lobe')
    pie_charts_ISPC(df, title)
    # df = df.groupby('img').mean()
    # df['BOLD_score'] = stats.zscore(df['BOLD_score'])
    # df['conn_score'] = stats.zscore(df['conn_score'])



if __name__ == '__main__':
    run_lmer_PFC_ISPC()
