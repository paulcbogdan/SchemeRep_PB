from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from connRSA.conn_analyze_IRAFs import ROI2NETWORK
from connRSA.conn_regress import send_to_specific, do_regr_RSA_sn, prep_ROI_avg, get_title
from connRSA.conn_utils import get_BNA_ROIs
from connRSA.single_trial_conn import prep_fps
from old.networks import prep_networks
from org_sns import get_sns
from utils import pickle_wrap

def get_stars(p):
    if p < .001:
        stars = '***'
    elif p < .01:
        stars = '**'
    elif p < .05:
        stars = '*'
    elif p < .1:
        stars = '†'
    else:
        stars = 'NS'
    return stars

def plot_beta_dif_bars(kwargs, corr=False):
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
            beta2, beta1, dif = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                            verbose=0, easy_override=True)
            if np.isnan(beta1):
                continue

            betas1_all[i, j] = beta1
            betas2_all[i, j] = beta2
            betas_dif_all[i, j] = dif
            # print(f'{beta1}, {beta2}, {dif}')

    # betas1_all = np.reshape(betas1_all, -1)[:, None]
    # betas2_all = np.reshape(betas2_all, -1)[:, None]
    # nans = np.isnan(betas1_all) | np.isnan(betas2_all)
    # betas1_all = betas1_all[~nans, None]
    # betas2_all = betas2_all[~nans, None]

    betas1_all *= 1000
    betas2_all *= 1000

    M_sns1 = np.nanmean(betas1_all, axis=1)
    M1 = np.nanmean(M_sns1)
    SE1 = np.nanstd(M_sns1) / np.sqrt(len(M_sns1))
    t1, p1 = stats.ttest_1samp(M_sns1, 0)
    M_sns2 = np.nanmean(betas2_all, axis=1)
    M2 = np.nanmean(M_sns2)
    SE2 = np.nanstd(M_sns2) / np.sqrt(len(M_sns2))
    t2, p2 = stats.ttest_1samp(M_sns2, 0)
    print(f'{t1=:.3f} {t2=:.3f}')

    # plt.scatter(M_sns1, M_sns2)
    # low = min(min(M_sns1), min(M_sns2))
    # high = max(max(M_sns1), max(M_sns2))
    # plt.plot([low, high], [low, high], color='k',)
    # plt.plot([0, 0], [low, high], color='k',)
    # plt.plot([low, high], [0, 0], color='k',)
    #

    # plt.show()
    # return

    fig = plt.figure(figsize=(4.2, 5))
    fontsize = 20
    plt.rcParams.update({'font.size': fontsize,
                         'font.sans-serif': 'Arial'})
    fig.subplots_adjust(bottom=0.18, left=0.24, right=0.85, top=0.85)
    bar_names = ['Local', 'Distributed', ]
    colors = ['dodgerblue', 'crimson', ]

    t_dif, p_dif = stats.ttest_rel(M_sns1, M_sns2)
    # t_dif, p_dif = stats.wilcoxon(M_sns1, M_sns2)

    M1_pos_p = np.nanmean(M_sns1 > 0)
    M2_pos_p = np.nanmean(M_sns2 > 0)
    dif_p = np.nanmean(M_sns1 > M_sns2)
    r, p = stats.pearsonr(M_sns1, M_sns2)


    # print(M)

    print(f'{t_dif=:.3f} (N = {len(M_sns1)}) {p_dif=:.3f} | '
          f'{M1_pos_p=:.1%} - {M2_pos_p=:.1%}: {dif_p:.1%} | '
          f'{r=:.3f}')
    # return
    plt.bar(bar_names, [M1, M2], yerr=[SE1, SE2],
            label='2', color=colors, capsize=5,
            linewidth=1., edgecolor='k')

    max_yerr = max(M1 + SE1, M2 + SE2)
    height = max_yerr * 1.05

    if p_dif < .10:
        plt.plot([0, 1], [height, height], color='k', linewidth=1.5)
        stars = get_stars(p_dif)
        stars_fs = 45 if stars != 'NS' else 24
        stars_height = height * .92 if stars != 'NS' else height * 1.02
        plt.text(0.5, stars_height, stars, ha='center', va='bottom',
                 fontsize=stars_fs)

    if M1 > 0 and p1 < .1:
        stars1 = get_stars(p1)
        # minus height accounts for asterisks not being centered in its box
        stars1_height = (M1 - SE1) * .5 - height * .053
        stars_fs = 45 if stars1 != 'NS' else 24
        plt.text(0, stars1_height, stars1, ha='center', va='center',
                 color='w', fontsize=stars_fs)
    if M2 > 0 and p2 < .1:
        stars2 = get_stars(p2)
        stars2_height = (M2 - SE2) * .5 - height * .053
        stars_fs = 45 if stars2 != 'NS' else 24
        plt.text(1, stars2_height, stars2, ha='center', va='center',
                 color='w', fontsize=stars_fs)


    plt.ylabel('Mean beta', fontsize=24, labelpad=10)

    plt.gca().spines[['top', 'right']].set_visible(False)
    title, fn, _ = get_title(True, False, kwargs, 28)
    plt.title(title, fontsize=15, pad=30)
    plt.ylim(0, height * 1.05)

    plt.yticks(range(0, int(height * 1.07) + 1,
                     min(max(int(height * 1.07) // 4, 1), 5)),
               fontsize=24)

    plt.xticks(bar_names, fontsize=20.5)

    plt.locator_params(axis='y', nbins=6)

    fn = f'beta_{kwargs["ROI_focus"]}_{fn}'

    fp_out = rf'result_pics/connRSA/RSA/{fn}'
    Path(fp_out).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fp_out, dpi=300)
    plt.show()


def do_regr_dif(semantic=False, RSA=True):
    trial_similarity = 'corr'
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    stdize_by_run = False
    regress_row = False

    target_ROIs = ['Occipital', 'ITL', 'Parietal', 'PFC']
    # target_ROIs = ['OC_IT']

    ts_ROI, ts_BOLD, ts_conn = [], [], []
    for target_ROI in target_ROIs:

        regions = set(prep_networks(
            network_setting=ROI2NETWORK[target_ROI])[target_ROI])
        ROIs_match = [ROI for region in regions
                          for ROI in get_BNA_ROIs() if region in ROI]
        ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]

        kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
                  'fp1': None, 'trial_similarity': trial_similarity,
                  'second_order': second_order,
                  'RDM_method': RDM_method,
                  'stdize_by_run': stdize_by_run,
                  'regress_row': regress_row, 'four_tasks': four_tasks,
                  }

        target_name = fr'{target_ROI}_M'
        prep_ROI_avg(target_name, ROI_lvl_control, RSA=RSA, ISPC=False,
                     ERS_alt=False, **kwargs)

        kwargs['ROI_focus'] = f'{target_ROI}_BOLD'
        kwargs['ROIs_ctrl'] = [f'{target_ROI}_M']
        plot_beta_dif_bars(kwargs)


import sys
sys.setrecursionlimit(10000)

if __name__ == '__main__':
    # do_regr_dif()
    do_regr_dif(semantic=True)

