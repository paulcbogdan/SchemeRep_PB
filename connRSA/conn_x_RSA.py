from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from tqdm.contrib.telegram import tqdm

from atlas_utils import get_atlas
from connRSA.conn_regress import do_regr_RSA_sn
from connRSA.information_connectivity import get_IC_mat, get_cross_IC_mat, get_cross_ERS_mat, get_cross_IRAF_mat
from connRSA.old.conn_x_RSA_lmer import get_idxs
from connRSA.single_trial_conn import prep_fps
from old.network_funcs import load_FC_for_Lifu
from org_sns import get_sns
from utils import pickle_wrap

import os
os.chdir(r'H:\PycharmProjects_H\SchemeRep')


def prep_conn_corrs(ERS=False):
    # semantic = False
    # regress_FC = False

    dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)


    # ERS = False
    ERS_nan_block = False

    cross = False
    IRAF = False

    semantic = False
    drop_con = False
    same_RSM_corr = False
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    second_order = 'spear'
    atlas = get_atlas()
    ROIs = atlas['ROIs']
    # print(f'{ROIs=}')
    # quit()
    if cross and ERS:
        assert not drop_con

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117',
           '118', '119', '120', '123', '124', '126', '127', '128', '129',
           '130', '131', '132', '134', '135', '136',
           '137', '138', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214',
           '216', '217', '218', '219', '221', '222', '224', '225', '227',
           '230', '232', '233', '234', '235', '239']

    four_tasks = '7'
    fps = prep_fps(four_tasks)

    kwargs = {'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'ROIs': ROIs,
              }

    corrs = []
    # TODO: maybe regress out the activation normal FC matrix?

    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]
    ERS_scores_all = []

    for i, sn in enumerate(sns):#, desc=f'Looping IC: {cross=}'):
        # print(f'Onto: {sn}')
        kwargs['sn'] = sn
        if IRAF:
            kwargs['fps'] = fps
            kwargs['second_order'] = 'spear'
            kwargs['RDM_method'] = 'within_nan'
            kwargs['semantic'] = semantic
            sn_corrs = pickle_wrap(get_cross_IRAF_mat,
                                   kwargs=kwargs, verbose=-1,
                                   easy_override=False,
                                   dt_max=dt_max)

        elif ERS:
            kwargs['fps'] = fps
            kwargs['cross'] = cross
            kwargs['nan_block'] = ERS_nan_block

            sn_corrs, ERS_scores = pickle_wrap(get_cross_ERS_mat,
                                               kwargs=kwargs, verbose=-1,
                                               easy_override=False,
                                               dt_max=dt_max)
            ERS_scores_all.append(ERS_scores)
        elif cross:
            kwargs['second_order'] = second_order
            kwargs['within_nan'] = True
            kwargs['fps'] = fps
            kwargs['same_RSM_corr'] = same_RSM_corr
            sn_corrs = pickle_wrap(get_cross_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False, dt_max=dt_max)
        else:
            kwargs['second_order'] = second_order
            kwargs['within_nan'] = True
            sn_corrs = []
            for fp in fps:
                kwargs['fp'] = fp
                corr = pickle_wrap(get_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False, dt_max=dt_max)
                sn_corrs.append(corr)
            sn_corrs = np.array(sn_corrs)

        # corr = np.nanmean(sn_corrs, axis=0)

        corrs.append(sn_corrs)

    corrs_all = np.array(corrs)
    # corrs_all = get_plain_corr()

    # print(corrs_all.shape)
    # quit()

    df_all = []
    for i in range(4):
        corrs = corrs_all[:, i]

        networks = ['Occipital', 'ITL', 'Parietal', 'PFC']
        network2name = {'Occipital': 'Occipital', 'ITL': 'Temporal',
                        'Parietal': 'Parietal', 'PFC': 'PFC'}
        names = [network2name[net] for net in networks]
        net2idxs = {}
        df_as_l = defaultdict(list)

        colors = ['dodgerblue', 'darkorange', 'crimson', 'limegreen']
        # plt.gcf().add_axes([0.1,0.1, 0.35,0.8])

        for net in networks:
            idxs = get_idxs(net)
            net2idxs[net] = idxs
            net_corr = corrs[:, idxs][:, :, idxs]
            net_sn_vals = np.nanmean(net_corr, axis=(1, 2))
            net_M = np.nanmean(net_sn_vals)
            net_SD = np.nanstd(net_sn_vals, ddof=1)
            net_SE = net_SD / np.sqrt(np.sum(~np.isnan(net_sn_vals)))
            df_as_l['net'].extend([network2name[net]]*len(sns))
            df_as_l['sn'].extend(sns)
            df_as_l['val'].extend(net_sn_vals)

        df = pd.DataFrame(df_as_l)
        df = df[df['net'] == 'Occipital']
        df['fp'] = fps[i]
        df_all.append(df)
    df = pd.concat(df_all)
    # print(df)
    return df

def get_RSA_betas():
    kwargs = {'semantic': True, 'fp': None, 'fp0': None,
              'fp1': None, 'trial_similarity': 'corr',
              'second_order': 'spear',
              'RDM_method': 'within_nan',
              'stdize_by_run': False,
              'regress_row': False, 'four_tasks': '7',
              'ROI_focus': 'OC_IT_BOLD', 'ROIs_ctrl': []#['ITL_M'],
              }

    fps = prep_fps(kwargs['four_tasks'])
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}

    betas1_all = np.full((len(sns), len(fps)), np.nan)
    betas2_all = np.full((len(sns), len(fps)), np.nan)
    betas_dif_all = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kwargs['cv'] = False
            kwargs['return_dif'] = True
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            if (sn, fp) in bad_tups:
                continue

            # if corr:
            #     beta2, beta1, dif = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
            #                                     verbose=-1)
            # else:
            beta1, beta2, dif = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                            verbose=-1, easy_override=False)
            if np.isnan(beta1):
                continue

            betas1_all[i, j] = beta1
            betas2_all[i, j] = beta2
            betas_dif_all[i, j] = dif
    # print(betas1_all.shape)
    # quit()
    df_all = []
    for i in range(4):
        df = pd.DataFrame({'sn': sns, 'beta1': betas1_all[:, i],
                           'fp': fps[i]})
        df_all.append(df)
    df = pd.concat(df_all)
    # print(df)
    return df

# def do_regress_FC():
#     for sn_i in range(corrs.shape[0]):
#         corr = corrs[sn_i]
#         corr_flat = corr.flatten()
#         corr_FC = corrs_FC[sn_i]
#         corr_FC_flat = corr_FC.flatten()
#         nans = np.isnan(corr_flat) | np.isnan(corr_FC_flat)
#         corr_flat_ = corr_flat[~nans]
#         corr_FC_flat_ = corr_FC_flat[~nans]
#         slope, intercept, r, p, se = (
#             stats.linregress(corr_FC_flat_, y=corr_flat_,
#                              alternative='two-sided'))
#         print(f'{slope=}')
#
#         corr_flat -= slope * corr_FC_flat
#         corr = corr_flat.reshape(corr.shape)
#         corrs[sn_i] = corr

def get_plain_corr():
    four_tasks = '7'
    fps = prep_fps(four_tasks)

    corrs = []
    sns = None
    prev_sns = None
    # fps = ['obj7_fMRI']
    for fp in fps:
        kwargs = {'fp': fp,
                  'split': False,
                  'key': 'inc',
                  'key_vals': (1, 2, 3),
                  'strict_sns': True,
                  'get_df_sn': True
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
            pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                        easy_override=False, verbose=1, cache_dir='cache',
                        RAM_cache=True)
        sns = [df_sn['sn'].iloc[0] for df_sn in df_sns]
        if prev_sns is None:
            prev_sns = sns
        else:
            assert tuple(prev_sns) == tuple(sns)
        corrs.append(sn_conn)
    return np.array(corrs).transpose((1, 0, 2, 3))

def corr_RSA_conn():
    df_conn = prep_conn_corrs()
    # print(len(df_conn))
    df_rsa = get_RSA_betas()
    # print(len(df_rsa))
    # quit()
    df = pd.merge(df_conn, df_rsa, on=['sn', 'fp'])
    # print(df)

    df = df.dropna()
    # df = df.sort_values('val')
    # print(df)
    # quit()

    from pymer4.models import Lmer
    formula = 'beta1 ~ val + (1|sn) '
    model = Lmer(formula, data=df)
    model.fit(summarize=False)
    print(model.summary())

    plt.scatter(df['val'], df['beta1'])
    plt.show()
    # r, p = stats.spearmanr(df['val'], df['beta1'])#, nan_policy='omit')
    # print(f'{r=:.3f}, {p=:.4f}')

if __name__ == '__main__':
    corr_RSA_conn()
    # DF = prep_conn_corrs(ERS=True)
    # get_RSA_betas()








