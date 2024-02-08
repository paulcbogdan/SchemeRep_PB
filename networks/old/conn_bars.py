import os
os.chdir('/')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from conn_utils import get_BNA_ROIs
from emotemporal_bar import run_ANOVA_ttest, setup_plots, add_interaction_lines
from old.network_clf import generic_prep
from utils import stdize
import matplotlib

def plot_bars(vals, title):
    df_YA = pd.DataFrame(np.array([vals[0], vals[1]]).T,
                         columns=['Inc', 'Con'])
    df_OA = pd.DataFrame(np.array([vals[2], vals[3]]).T,
                         columns=[' Inc ', ' Con '])
    df_scores = pd.concat([df_YA, df_OA], axis=0)
    X = list(df_YA.columns) + list(df_OA.columns)

    font = {'size': 16.5}
    matplotlib.rc('font', **font)
    M = df_scores[X].mean()
    SE = df_scores[X].std() / np.sqrt(df_scores[X].shape[0])
    lowest = M - SE * 1.2
    y_low = min(lowest)
    y_low = min(y_low, 0)
    y_high = max(M + SE) * 1.2
    # print(f'{y_low=}, {y_high=}')
    y_high = max(y_high, 0.07)
    axs = setup_plots(False)
    X_labels = X
    colors = ['red', 'dodgerblue', 'red', 'dodgerblue']
    plt.bar(X_labels, M, yerr=SE, color=colors, capsize=5)
    # plt.ylabel(key_to_label[key], fontsize=17.5)
    plt.yticks(fontsize=15)
    upper_signif_line, lower_signif_line = \
        add_interaction_lines(y_high, y_low)
    stars_itr, highest_line = run_ANOVA_ttest(df_scores, X, y_high, y_low,
                                              upper_signif_line)

    plt.gca().spines[['right', 'top']].set_visible(False)

    # Center x-labeled
    plt.text(0.5, y_low - (y_high - y_low) * .2, 'Younger Adults',
             fontsize=17.5, ha='center')

    plt.text(2.5, y_low - (y_high - y_low) * .2, 'Older adults',
             fontsize=17.5, ha='center')

    line = plt.Line2D([0.5, 0.5], [-0.003, -.21],
                      transform=plt.gca().transAxes,
                      color='black',
                      dash_capstyle='butt')
    line.set_clip_on(False)
    plt.gca().add_line(line)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True)

    plt.ylim(bottom=y_low, top=highest_line)
    plt.title(title, fontsize=14)
    plt.ylabel('Connectivity')
    plt.show()


def run_IPL_bars(threshold=0.95):
    fp = 'obj4_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'odd_even': False,
              }
    partitions, sn_inc_act, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)

    ROIs = get_BNA_ROIs()
    region0 = 'ATL'
    idxs0 = [i for i, ROI in enumerate(ROIs) if region0 in ROI]
    regions1 = ['sOcG', 'EVC', 'LOC', 'FuG',
                'ITG', 'PhG', 'MTG', 'Hipp', 'IPL',
                'IFG', 'MFG', 'SFG', 'OrG']
    # regions1 = ['IFG']
    for region1 in regions1:
        idxs1 = [i for i, ROI in enumerate(ROIs) if region1 in ROI]
        vals = []
        for age in [1, 2]:
            print(sn_inc_act.shape)
            print(idxs0)
            sn_inc_act_age = sn_inc_act[age2idxs[age]]
            sn_inc_act0 = sn_inc_act_age[:, :, idxs0, :]
            sn_inc_act0 = stdize(sn_inc_act0, axis=3, nans=True)

            sn_inc_act1 = sn_inc_act_age[:, :, idxs1, :]
            sn_inc_act1 = stdize(sn_inc_act1, axis=3, nans=True)

            conn = sn_inc_act0[..., None, :] * sn_inc_act1[..., None, :, :]
            conn = np.nanmean(conn, axis=-1)
            conn_M = np.nanmean(conn, axis=(-1, -2))

            cond0_M = conn_M[:, 0]
            cond1_M = conn_M[:, 1]
            vals += [cond0_M, cond1_M]
        title = f'{region0} x {region1}'
        plot_bars(vals, title)

        Ms = np.array([np.nanmean(cond_M) for cond_M in vals])
        SDs = np.array([np.nanstd(cond_M) for cond_M in vals])
        Ns = np.array([np.sum(~np.isnan(cond_M)) for cond_M in vals])
        SEs = SDs / np.sqrt(Ns)

def conn_a_lot(threshold=.95):
    fp = 'obj4_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'odd_even': False,
              }
    partitions, sn_inc_act, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)
    ROIs = get_BNA_ROIs()

    # regions0 = ['IPL', 'pSTS']
    regions0 = ['MFG', 'IFG']
    # regions0 = ['EVC', 'LOC']
    regions1 = ['ATL']

    # regions1 = ['EVC', 'LOC', 'FuG', 'ITG', 'ATL']
    idxs0 = []
    for r0 in regions0:
        idxs0_r = [i for i, ROI in enumerate(ROIs) if r0 in ROI]
        idxs0 += idxs0_r
    idxs1 = []
    for r1 in regions1:
        idxs1_r = [i for i, ROI in enumerate(ROIs) if r1 in ROI]
        idxs1 += idxs1_r

    vals = []
    for age in [1, 2]:
        sn_inc_act_age = sn_inc_act[age2idxs[age]]
        sn_inc_act0 = sn_inc_act_age[:, :, idxs0, :]
        print(f'{sn_inc_act0.shape=}')
        sn_inc_act0 = stdize(sn_inc_act0, axis=3, nans=True)

        sn_inc_act1 = sn_inc_act_age[:, :, idxs1, :]
        print(f'{sn_inc_act1.shape=}')
        sn_inc_act1 = stdize(sn_inc_act1, axis=3, nans=True)

        conn = sn_inc_act0[..., None, :] * sn_inc_act1[..., None, :, :]
        conn = np.nanmean(conn, axis=-1)
        conn_M = np.nanmean(conn, axis=(-1, -2))

        cond0_M = conn_M[:, 0]
        cond1_M = conn_M[:, 1]
        vals += [cond0_M, cond1_M]
    title = f'{regions0} x\n {regions1}'
    plot_bars(vals, title)
    quit()

    Ms = np.array([np.nanmean(cond_M) for cond_M in vals])
    SDs = np.array([np.nanstd(cond_M) for cond_M in vals])
    Ns = np.array([np.sum(~np.isnan(cond_M)) for cond_M in vals])
    SEs = SDs / np.sqrt(Ns)


if __name__ == '__main__':
    # run_IPL_bars()
    conn_a_lot()
