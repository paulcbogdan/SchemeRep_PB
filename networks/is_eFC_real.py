from tqdm import tqdm

from HCP import load_HCP_act
from analyze_rs import load_act_conn, prep_conn_ps
from atlas_utils import get_atlas
from utils import pickle_wrap
from collections import defaultdict

from vendor_lmers import get_module_trialwise_z, get_dfs_conn_trials, get_module_cross_trialwise_z
from scipy import stats
import pandas as pd
import numpy as np
import pingouin as pg

import matplotlib.pyplot as plt
from vendor_partitioning import get_vendor_partitions
from analyze_rs import load_resting_data
from functools import partial
import os
import statsmodels.formula.api as smf


def test_reality(fp='HCP_RS', norm_std=True, alpha=.01):
    if fp == 'HCP_RS':
        sn_roi_act, sns = pickle_wrap(load_HCP_act, kwargs={'N': 5,
                                                            'RS': True})
        # print(f'{sn_roi_act.shape=}')
        # quit()
    elif fp == 'rs':
        sn_roi_act, sns, conn_trials = load_act_conn(norm_std,
                                                     easy_override=False,
                                                     )
    else:
        sn_roi_act, conn_trials, sns = \
                pickle_wrap(get_dfs_conn_trials, kwargs={'fp': fp,
                                                         'single': False,
                                                         'w_activity': True,
                                                         'squeeze': True})
    # print(sn_roi_act.shape)
    # print(f'{len(sns)=}')
    # quit()
    n_roi = sn_roi_act.shape[1]
    n_sn = sn_roi_act.shape[0]
    ps = []
    ps_basic = []
    cnt = defaultdict(lambda: 0)
    cnt_sim = defaultdict(lambda: 0)
    nsim = 10_000
    prop_var_explained = []
    r_basics = []
    itr_ests = []
    r_sims = []
    itr_sims = []
    r_sims2 = []

    p_sims = []
    p_sims_itr = []
    bad_i = [8, 14, 15, 26, 33, 60]
    for _ in tqdm(range(nsim), desc='sim eFC'):
        a, b, c, d = np.random.choice(n_roi, 4, replace=False)
        a, b, c, d = np.sort([a, b, c, d])
        i = np.random.choice(n_sn)
        if fp == 'rs' and (i in bad_i): continue
        roi_act = sn_roi_act[i, :, :]


        df = pd.DataFrame({'a': roi_act[a, :], 'b': roi_act[b, :],
                           'c': roi_act[c, :], 'd': roi_act[d, :]})

        # cov = np.zeros((4, 4))
        # cov[np.diag_indices_from(cov)] = 1
        # cov[0, 3] = 0.5
        # cov[3, 0] = 0.5



        df.dropna(inplace=True)
        if len(df) < 10:
            continue

        for col in df.columns:
            df[col] = stats.zscore(df[col])
        df['ab'] = df['a'] * df['b']
        df['cd'] = df['c'] * df['d']

        # formula = 'ab ~ 1 + cd + a * c + a * d + b * d'
        formula = 'a ~ 1 + b * c * d'
        model = smf.ols(formula=formula, data=df)
        res = model.fit()

        key = 'b:c:d'
        p = res.pvalues[key]
        est_eFC = res.params[key]
        # print(f'{i}, {est_eFC=:.3f}')
        # if est_eFC > 1.0:
        #     continue
        ps.append(p)
        itr_ests.append(est_eFC)

        var_eFC = res.params[key] ** 2
        var_nFC = 0
        # for x in ['b', 'c', 'd']:
        #     var_nFC += res.params[x] ** 2
        prop_var_explained.append(var_eFC / (var_eFC + var_nFC))

        r_basic, p_basic = stats.pearsonr(df['ab'], df['cd'])
        if abs(r_basic) > .999:
            print(f'{i=}, {a}, {b}, {c}, {d}')
            # quit()
        ps_basic.append(p_basic)
        r_basics.append(r_basic)
        cnt[(p < alpha, p_basic < alpha)] += 1

        cov = df[['a', 'b', 'c', 'd']].corr()
        ar = np.random.multivariate_normal([0, 0, 0, 0], cov,
                                           10_000)
        df_sim = pd.DataFrame(ar, columns=['a', 'b', 'c', 'd'])
        df_sim['ab'] = df_sim['a'] * df_sim['b']
        df_sim['cd'] = df_sim['c'] * df_sim['d']
        r_sim, p_sim = stats.pearsonr(df_sim['ab'], df_sim['cd'])
        r_sims.append(r_sim)
        p_sims.append(p_sim)

        formula = 'a ~ 1 + b * c * d'
        model = smf.ols(formula=formula, data=df_sim)
        res = model.fit()
        key = 'b:c:d'
        p_sim_itr = res.pvalues[key]
        est_eFC = res.params[key]
        itr_sims.append(est_eFC)
        p_sims_itr.append(p_sim_itr)


        ar2 = np.random.multivariate_normal([0, 0, 0, 0], cov,
                                           len(df))
        df_sim2 = pd.DataFrame(ar2, columns=['a', 'b', 'c', 'd'])
        df_sim2['ab'] = df_sim2['a'] * df_sim2['b']
        df_sim2['cd'] = df_sim2['c'] * df_sim2['d']
        r_sim2, _ = stats.pearsonr(df_sim2['ab'], df_sim2['cd'])
        r_sims2.append(r_sim2)

        cnt_sim[(p_sim_itr < alpha, p_sim < alpha)] += 1


    ps = np.array(ps)
    ps_basic = np.array(ps_basic)
    plt.title('Interaction')
    plt.hist(ps, bins=100, color='dodgerblue')
    plt.show()
    plt.hist(ps_basic, bins=100, color='r')
    plt.title('Basic')
    plt.show()

    both_sig = cnt[(True, True)]
    proper_sig = cnt[(True, False)]
    basic_sig = cnt[(False, True)]
    neither_sig = cnt[(False, False)]

    print(f'{both_sig=}, {proper_sig=}, {basic_sig=}, {neither_sig=}')

    prop_var_explained = np.array(prop_var_explained)
    prop_var_explained = prop_var_explained[ps < .01]
    print(f'{np.mean(prop_var_explained)=:.3f} '
          f'{np.std(prop_var_explained)=:.3f}')
    print(f'{len(prop_var_explained)=}')




    # plt.hist(prop_var_explained, bins=100)
    # plt.xlabel('fluctuation var explained')
    # plt.show()
    # itr_ests = np.array(itr_ests)[ps < .01]
    # r_sims = np.array(r_sims)[ps < .01]

    itr_ests = np.array(itr_ests)
    r_basics = np.array(r_basics)
    r_sims = np.array(r_sims)
    r_sims2 = np.array(r_sims2)

    itr_sigs = ps < alpha
    r_sigs = ps_basic < alpha

    itr_only = np.logical_and(itr_sigs, ~r_sigs)
    r_only = np.logical_and(~itr_sigs, r_sigs)
    both = np.logical_and(itr_sigs, r_sigs)
    neither = np.logical_and(~itr_sigs, ~r_sigs)
    label2excl = {'itr_only': itr_only, 'r_only': r_only,
                  'both': both, 'neither': neither}
    label2corr = {}
    for label, excl in label2excl.items():
        print(f'{label=}')
        r, _ = stats.pearsonr(r_basics[excl], r_sims[excl])
        r2, _ = stats.pearsonr(r_sims[excl], r_sims2[excl])
        label2corr[label] = f'\n(r = {r:.2f} / {r2:.2f})'

    l = itr_ests
    plt.scatter(l[neither], r_basics[neither], color='k',
                label='Neither' + label2corr['neither'], alpha=.05)
    plt.scatter(l[r_only], r_basics[r_only], color='r',
                label='Correlation only' + label2corr['r_only'], alpha=.1)
    plt.scatter(l[both], r_basics[both], color='g',
                label='Both' + label2corr['both'], alpha=.3)
    plt.scatter(l[itr_only], r_basics[itr_only], color='dodgerblue',
                label='Interaction only' + label2corr['itr_only'], alpha=.3)
    plt.legend(frameon=False)
    plt.title(f'{both_sig=}, {proper_sig=},\n'
              f'{basic_sig=}, {neither_sig=}')
    plt.ylabel('edge-edge correlation')
    plt.xlabel('node-node-node-node interaction')
    plt.show()

    p_sims_itr = np.array(p_sims_itr)
    p_sims = np.array(p_sims)
    itr_sim_sigs = p_sims_itr < alpha
    r_sim_sigs = p_sims < alpha

    itr_only = np.logical_and(itr_sim_sigs, ~r_sim_sigs)
    r_only = np.logical_and(~itr_sim_sigs, r_sim_sigs)
    both = np.logical_and(itr_sim_sigs, r_sim_sigs)
    neither = np.logical_and(~itr_sim_sigs, ~r_sim_sigs)

    l = r_sims
    plt.scatter(l[neither], r_basics[neither], color='k',
                label='Neither' + label2corr['neither'], alpha=.05)
    plt.scatter(l[r_only], r_basics[r_only], color='r',
                label='Correlation only' + label2corr['r_only'], alpha=.05)
    plt.scatter(l[both], r_basics[both], color='g',
                label='Both' + label2corr['both'], alpha=.05)
    plt.scatter(l[itr_only], r_basics[itr_only], color='dodgerblue',
                label='Interaction only' + label2corr['itr_only'], alpha=.05)
    plt.legend(frameon=False)

    both_sig = cnt_sim[(True, True)]
    proper_sig = cnt_sim[(True, False)]
    basic_sig = cnt_sim[(False, True)]
    neither_sig = cnt_sim[(False, False)]
    plt.title(f'{both_sig=}, {proper_sig=},\n'
              f'{basic_sig=}, {neither_sig=}')

    plt.ylabel('edge-edge correlation')
    plt.xlabel('edge-edge simulated')
    plt.show()
    quit()





    r, p = stats.spearmanr(itr_ests, r_basics)
    print(f'{r=:.3f}, {p=:.3f}')

    itr_ests_ = np.array(itr_ests)[ps < .05]
    r_basics_ = np.array(r_basics)[ps < .05]
    r, p = stats.spearmanr(itr_ests_, r_basics_)
    print(f'Threshold to p < .05: {r=:.3f}, {p=:.3f}')

    itr_ests_ = np.array(itr_ests)[ps < .01]
    r_basics_ = np.array(r_basics)[ps < .01]
    r, p = stats.spearmanr(itr_ests_, r_basics_)
    print(f'Threshold to p < .01: {r=:.3f}, {p=:.3f}')

    # plt.scatter(1/ps_basic, 1/ps)
    # plt.ylabel('p-values (interaction)')
    # plt.xlabel('p-values (basic)')
    # plt.show()



if __name__ == '__main__':
    test_reality()






