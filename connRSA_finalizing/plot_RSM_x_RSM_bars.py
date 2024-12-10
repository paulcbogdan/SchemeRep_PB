import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.conn_regress import do_regr_RSA_sn, prep_ROI_avg, get_title
from Utils.atlas_funcs import get_BNA_ROIs
from connRSA.single_trial_conn import prep_fps
from networks.old.networks import prep_networks
# from old.networks import prep_networks
from org_sns import get_sns
from Utils.pickle_wrap_funcs import pickle_wrap
import pathlib
path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)
#
# os.chdir(r'/')


def get_stars(p, no_cross=True):
    if p < .001:
        stars = '***'
    elif p < .01:
        stars = '**'
    elif p < .05:
        stars = '*'
    elif p < .1 and not no_cross:
        stars = '†'
    else:
        stars = 'NS'
    return stars

def interaction_bars(kwargs):
    fps = prep_fps(kwargs['four_tasks'])
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}

    conds = [('Occipital', 'Local'), ('Occipital', 'Distributed'),
             ('IT', 'Local'), ('IT', 'Distributed')]
    cond2betas = {}
    for cond in conds:
        cond2betas[cond] = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kwargs['cv'] = False
            kwargs['return_dif'] = True
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            if (sn, fp) in bad_tups:
                continue

            for cond in conds:
                if cond[1] == 'Local':
                    kwargs['ROI_focus'] = f'{cond[0]}_M'
                else:
                    kwargs['ROI_focus'] = f'{cond[0]}_BOLD'
                kwargs['ROIs_ctrl'] = []
                # print(f'{kwargs=}')
                # quit()
                beta1, _, _ = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                          verbose=-1, easy_override=False)

                cond2betas[cond][i, j] = beta1 * 1000

    cond2betas = {cond: np.nanmean(betas, axis=1) for cond, betas
                  in cond2betas.items()}

    cond2Ms = {cond: np.nanmean(betas) for cond, betas in cond2betas.items()}
    # cond2SEs = {cond: np.nanstd(betas) / np.sqrt(len(betas))
    #             for cond, betas in cond2betas.items()}
    cond2SEs = {cond: np.nanstd(betas, ddof=1) / np.sqrt(len(betas))
                for cond, betas in cond2betas.items()}

    itr = (cond2betas[conds[0]] - cond2betas[conds[1]] -
           cond2betas[conds[2]] + cond2betas[conds[3]])
    t, p = stats.ttest_1samp(itr, 0)

    oc_ef = cond2betas[conds[0]] - cond2betas[conds[1]]
    oc_t, oc_p = stats.ttest_1samp(oc_ef, 0)
    ITL_ef = cond2betas[conds[2]] - cond2betas[conds[3]]
    ITL_t, ITL_p = stats.ttest_1samp(ITL_ef, 0)

    print(f'Interaction: {t=:.3f}, {p=:.3f}')
    print(f'\tOccipital: {oc_t=:.3f}, {oc_p=:.3f}')
    print(f'\tITL: {ITL_t=:.3f}, {ITL_p=:.3f}')

    Ms = np.array([cond2Ms[cond] for cond in conds])
    SEs = np.array([cond2SEs[cond] for cond in conds])
    max_yerr = max(Ms + SEs)
    height = max_yerr * 1.05

    plt.rcParams.update({'font.sans-serif': 'Arial'})
    colors = ['dodgerblue' if 'Local' in cond[1] else 'orange'
              for cond in conds]
    plt.bar([0, 1, 2, 3], Ms, yerr=SEs, color=colors,
            capsize=5, linewidth=1., edgecolor='k')
    plt.xticks([0, 1, 2, 3],
               ['Local', 'Distributed\n(Voxelwise)',
                'Local', 'Distributed\n(Voxelwise)'],
               fontsize=19)

    # plt.text(0.5, 0 - height * .19, 'Occipital',
    #          fontsize=22, ha='center')
    # plt.text(2.5, 0 - height * .19, 'Temporal',
    #          fontsize=22, ha='center')

    plt.text(0.5, 0 - height * .32, 'Occipital',
             fontsize=22, ha='center')
    plt.text(2.5, 0 - height * .32, 'Temporal',
             fontsize=22, ha='center')

    lower_signif_line = max_yerr * 1.04
    upper_signif_line = max_yerr * 1.1
    plt.plot([0, 1], [lower_signif_line, lower_signif_line], color='k')
    plt.plot([0.5, 0.5], [lower_signif_line, upper_signif_line], color='k')
    plt.plot([2, 3], [lower_signif_line, lower_signif_line], color='k')
    plt.plot([2.5, 2.5], [lower_signif_line, upper_signif_line], color='k')
    plt.plot([0.5, 2.5], [upper_signif_line, upper_signif_line], color='k')

    # stars_height = height * .92 if stars != 'NS' else height * 1.02
    itr_stars = get_stars(p)
    itr_stars = itr_stars.replace('NS', ' ')
    stars_fs = 35 if itr_stars not in ['NS', '†'] else 24
    # if 'NS' not in itr_stars:
    plt.text(1.5,
             max_yerr*1.12 if itr_stars not in ['NS', '†'] else
             max_yerr*1.19,
             itr_stars, ha='center', va='center',
             fontsize=stars_fs, )

    oc_stars = get_stars(oc_p)
    if 'NS' not in oc_stars:
        stars_fs = 35
        plt.text(0.5, max_yerr*.935, oc_stars, ha='center', va='center',
                 fontsize=stars_fs, )
    itl_stars = get_stars(ITL_p)
    if 'NS' not in itl_stars:
        stars_fs = 35
        plt.text(2.5, max_yerr * .935, itl_stars, ha='center', va='center',
                 fontsize=stars_fs, )


    # plt.yticks(range(0, int(height * 1.07) + 1,
    #                  min(max(int(height * 1.07) // 4, 1), 5)),
    #            fontsize=24)
    plt.locator_params(axis='y', nbins=6)

    plt.ylabel('Mean correlation', fontsize=24, labelpad=10)
    plt.gca().spines[['top', 'right', 'bottom']].set_visible(False)
    plt.ylim(0, height * 1.07)

    # line = plt.Line2D([0.5, 0.5], [-.003, -0.18],
    #                   transform=plt.gca().transAxes,
    #                   color='black', linewidth=1.,
    #                   dash_capstyle='butt')
    # line.set_clip_on(False)
    # plt.gca().add_line(line)

    plt.tight_layout()


    plt.show()


def plot_beta_dif_bars(kwargs, corr=False, big_voxel=False,
                       drop_partial_bad_runs=False):
    fps = prep_fps(kwargs['four_tasks'])

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}
    # '132' # has corrupted run 3
    # '138' # Bad retrieval session for run 3
    # '224' # Was not able to finish the last run of encoding
    # '234' # Was not able to finish the first run of encoding

    # sns = ['132', '138', '224', '224']

    betas1_all = np.full((len(sns), len(fps)), np.nan)
    betas2_all = np.full((len(sns), len(fps)), np.nan)
    betas_dif_all = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kwargs['cv'] = False
            kwargs['return_dif'] = True
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            if drop_partial_bad_runs:
                if (sn, fp) in bad_tups:
                    continue

            beta1, beta2, dif = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                            verbose=-1, easy_override=True)
            if np.isnan(beta1):
                continue

            if beta2 is None:
                if '_M' in kwargs['ROI_focus']:
                    beta2 = beta1
                    beta1 = 0
                else:
                    beta2 = 0

            betas1_all[i, j] = beta1
            betas2_all[i, j] = beta2
            betas_dif_all[i, j] = dif

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
    print(f'{t1=:.3f}, {M1=:.2f} | {t2=:.3f}, {M2=:.2f}')

    fig = plt.figure(figsize=(4.2, 5))
    fontsize = 20
    plt.rcParams.update({'font.size': fontsize,
                         'font.sans-serif': 'Arial'})
    fig.subplots_adjust(bottom=0.18, left=0.24, right=0.85, top=0.85)
    bar_names = ['Local', 'Distributed', ]
    colors = ['dodgerblue', 'crimson', ]

    t_dif, p_dif = stats.ttest_rel(M_sns1, M_sns2, nan_policy='omit')
    # t_dif, p_dif = stats.wilcoxon(M_sns1, M_sns2)

    M1_pos_p = np.nanmean(M_sns1 > 0)
    M2_pos_p = np.nanmean(M_sns2 > 0)
    dif_p = np.nanmean(M_sns1 > M_sns2)



    print(f'{t_dif=:.3f} (N = {len(M_sns1)}) {p_dif=:.3f} | '
          f'{M1_pos_p=:.1%} - {M2_pos_p=:.1%}: {dif_p:.1%} | '
          f'') # {r=:.3f}
    # r, p = stats.pearsonr(M_sns1, M_sns2)
    # print(f'{r=:.3f}')
    # return
    plt.bar(bar_names, [M2, M1], yerr=[SE2, SE1],
            label='2', color=colors, capsize=5,
            linewidth=1., edgecolor='k')

    max_yerr = max(M1 + SE1, M2 + SE2)
    height = max_yerr * 1.05

    if p_dif < .10:
        plt.plot([0, 1], [height, height], color='k', linewidth=1.5)
        stars = get_stars(p_dif)
        stars_fs = 45 if stars not in ['NS', '†'] else 24
        stars_height = height * .92 if stars != 'NS' else height * 1.02
        plt.text(0.5, stars_height, stars, ha='center', va='bottom',
                 fontsize=stars_fs)

    if M1 > 0 and p1 < .1:
        stars1 = get_stars(p1)
        # minus height accounts for asterisks not being centered in its box
        stars1_height = (M1 - SE1) * .5 - height * .053
        stars_fs = 45 if stars1 not in ['NS', '†']  else 24
        plt.text(1, stars1_height, stars1, ha='center', va='center',
                 color='w', fontsize=stars_fs)
    if M2 > 0 and p2 < .1:
        stars2 = get_stars(p2)
        # print(f'{p2=:.4f}, {p1=:.4f}')
        stars2_height = (M2 - SE2) * .5 - height * .053
        stars_fs = 45 if stars2 not in ['NS', '†'] else 24
        plt.text(0, stars2_height, stars2, ha='center', va='center',
                 color='w', fontsize=stars_fs)


    plt.ylabel('Mean beta', fontsize=24, labelpad=10)
    plt.gca().spines[['top', 'right', 'bottom']].set_visible(False)
    title, fn, _ = get_title(True, False, kwargs, 28)
    plt.title(title, fontsize=15, pad=30)


    plt.locator_params(axis='y', nbins=6)

    plt.xticks(bar_names, fontsize=20.5)
    plt.ylim(0, height * 1.05)


    fn = f'beta_{kwargs["ROI_focus"]}_{fn}'

    fp_out = rf'result_pics/connRSA/RSA/{fn}'
    Path(fp_out).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fp_out, dpi=300)
    plt.show()


def do_regr_dif(semantic=False, RSA=True, big_voxelwise=True,
                inter=True, strict_corr=True):
    trial_similarity = 'corr'
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    stdize_by_run = False
    regress_row = False

    target_ROIs = ['Occipital', 'IT', 'Parietal', 'PFC']
    # target_ROIs = ['FP', 'DMN', 'FPT']
    # target_ROIs = ['IT']
    target_ROIs = ['ITL']
    # target_ROIs = ['Occipital']

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
                  'regress_row': regress_row,
                  'four_tasks': four_tasks,
                  }

        target_name = fr'{target_ROI}_M'
        prep_ROI_avg(target_name, ROI_lvl_control, RSA=RSA, ISPC=False,
                     ERS_alt=False, **kwargs)

        kwargs['ROI_focus'] = f'{target_ROI}_BOLD'

        if big_voxelwise:
            kwargs['ROI_focus'] += '_cmb'
        # kwargs['ROI_focus'] = f'{target_ROI}_M'

        if inter:
            kwargs['ROIs_ctrl'] = []
            interaction_bars(kwargs)
            return
        elif strict_corr:
            kwargs['ROIs_ctrl'] = ROI_lvl_control
            strict_correlations(kwargs)
            return
        kwargs['ROIs_ctrl'] = [f'{target_ROI}_M'] # + ROI_lvl_control[1:]
        plot_beta_dif_bars(kwargs)

def strict_correlations(kwargs):
    fps = prep_fps(kwargs['four_tasks'])
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}

    conds = [('Occipital', 'Average'), ('Occipital', 'Voxel'),
             ('IT', 'Average'), ('IT', 'Voxel')]
    cond2betas = {}
    for cond in conds:
        cond2betas[cond] = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kwargs['cv'] = False
            kwargs['return_dif'] = True
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            if (sn, fp) in bad_tups:
                continue

            for cond in conds:
                if cond[1] == 'Average':
                    kwargs['ROI_focus'] = f'{cond[0]}_BOLD'
                else:
                    kwargs['ROI_focus'] = f'{cond[0]}_BOLD_cmb'
                beta1, _, _ = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                          verbose=-1, easy_override=False)

                cond2betas[cond][i, j] = beta1 * 1000

    cond2betas = {cond: np.nanmean(betas, axis=1) for cond, betas
                  in cond2betas.items()}

    cond2Ms = {cond: np.nanmean(betas) for cond, betas in cond2betas.items()}
    print(cond2Ms)
    cond2SEs = {cond: np.nanstd(betas) / np.sqrt(len(betas))
                for cond, betas in cond2betas.items()}

    for cond, betas in cond2betas.items():
        t, p = stats.ttest_1samp(betas, 0)
        print(f'{cond=}, {t=:.3f}, {p=:.3f}')
    return

def plot_boxen(cond2betas, big_voxelwise):
    import pandas as pd
    import seaborn as sns

    all_betas = np.array([cond2betas[cond_other] for cond_other in cond2betas])
    height = np.quantile(all_betas, .98)
    floor = np.quantile(all_betas, .02)
    sns_out = np.any(all_betas > height, axis=0) | np.any(all_betas < floor, axis=0)
    cond2betas_ = {}
    for cond, betas in cond2betas.items():
        cond2betas_[cond] = betas[~sns_out]
    cond2betas = cond2betas_

    n_sn = len(cond2betas[(False, 'Local')])

    df_small_vis = pd.DataFrame({'model': ['Perceptual'] * n_sn,
                                 'small_large': ['Small'] * n_sn,
                                 'vals': cond2betas[(False, 'Local')]})
    df_large_vis = pd.DataFrame({'model': ['Perceptual'] * n_sn,
                                 'small_large': ['Large'] * n_sn,
                                 'vals': cond2betas[(False, 'Distributed')]})

    df_small_sem = pd.DataFrame({'model': ['Semantic'] * n_sn,
                                 'small_large': ['Small'] * n_sn,
                                 'vals': cond2betas[(True, 'Local')]})
    df_large_sem = pd.DataFrame({'model': ['Semantic'] * n_sn,
                                 'small_large': ['Large'] * n_sn,
                                 'vals': cond2betas[(True, 'Distributed')]})
    df = pd.concat([df_small_vis, df_large_vis, df_small_sem, df_large_sem],)


    cond2spot = {(False, 'Local'): 0, (False, 'Distributed'): 1,
                 (True, 'Local'): 2, (True, 'Distributed'): 3}
    cond2color = {(False, 'Local'): 'dodgerblue',
                  (False, 'Distributed'): 'orange' if big_voxelwise else 'red',
                  (True, 'Local'): 'dodgerblue',
                  (True, 'Distributed'): 'orange' if big_voxelwise else 'red'}
    cond2jitter = {}
    # height = -999.
    # floor = 999.


    # print(sns_above)
    # print(sns_above.shape)
    # quit()



    #

    for cond, spot in cond2spot.items():
        v = cond2betas[cond]
        # other = [cond2betas[cond_other] for cond_other in cond2spot]
        # M_all = np.mean(other, axis=0)
        # v_ = v - M_all
        # v = v_
        # print(np.mean(v))


        # jitter = np.random.uniform(-0.1, 0.1, len(v))
        jitter = np.random.uniform(-0.17, 0.17, len(v))
        dist = np.abs((v - np.mean(v)) / np.std(v))
        dist = np.clip(dist, 0, 3)
        dist = (3 - dist) / 3
        jitter *= dist
        cond2jitter[cond] = jitter
        # color2 = 'brown' if cond2color[cond] == 'orange' else 'navy'
        plt.scatter(np.full(len(v), spot) + jitter, v, color=cond2color[cond],
                    # color=color2,
                    s=25, alpha=0.25,
                    # facecolors='none',
                    linewidth=1,
                    #edgecolors='k'
                    )
        plt.scatter(np.full(len(v), spot) + jitter, v, color=cond2color[cond],
                    s=25, alpha=0.5,
                    facecolors='none',
                    linewidth=1,
                    )

        # if True:
        #     height = np.quantile(v, .97) if np.quantile(v, .97) > height else height
        #     floor = np.quantile(v, .03) if np.quantile(v, .03) < floor else floor
        # else:
        #     height = np.max(v) if np.max(v) > height else height
        #     floor = np.min(v) if np.max(v) < floor else floor


        M = np.mean(v)
        # plt.plot([spot - 0.1, spot + 0.1], [M, M], color='k', linewidth=1.5)
    # quit()
    for i in range(n_sn):
        for bl in [False, True]:
            plt.plot([cond2spot[(bl, 'Local')] +
                      cond2jitter[(bl, 'Local')][i],
                      cond2spot[(bl, 'Distributed')] +
                      cond2jitter[(bl, 'Distributed')][i],],
                     [cond2betas[(bl, 'Local')][i],
                      cond2betas[(bl, 'Distributed')][i]],
                     color='k', alpha=0.25,
                     linestyle=(5, (10, 3)),
                     linewidth=0.35, zorder=-1)



    plt.xticks([0.1, 0.9, 2.1, 2.9],
               ['Mean\nsmall', 'Large' if big_voxelwise else 'Large\n(Averages)',
                'Mean\nsmall', 'Large' if big_voxelwise else 'Large\n(Averages)'
                ],
               fontsize=19)
    plt.plot([-.5, 3.5], [0, 0], linewidth=0.5, color='k')
    plt.xlim(-.5, 3.5)
    pad = (height - floor) * .03
    return height, floor, pad

    # pad = (height - floor) * .03
    # plt.ylim(floor - pad, height + pad)
    # plt.gca().spines[['top', 'right']].set_visible(False)
    # plt.text(0.5, floor - pad * 12, 'Perceptual',
    #          fontsize=22, ha='center')
    # plt.text(2.5, floor - pad * 12, 'Semantic',
    #          fontsize=22, ha='center')
    # plt.subplots_adjust(bottom=0.24)  # Increases bottom margin to accommodate text

    plt.show()
    # quit()


    g = sns.catplot(x='model', y='vals', data=df,
                    hue='small_large',
                    kind='boxen',
                    # linecolor='k',
                    palette=['dodgerblue', 'orange'],
                    legend=False,
                    # saturation=0.9,
                    height=5, aspect=0.75,
                    # flier_kws={'edgecolor': ['k'],
                    #            'linewidth': 0.8,
                    #            'marker': '.',
                    #            },
                    )
    plt.show()
    quit()

    # g = sns.barplot(x='model', y='vals', data=df,
    #             hue='small_large',
    #             palette=['dodgerblue', 'orange'],
    #             capsize=0.1,
    #             errwidth=1.5,
    #             edgecolor='k',
    #             linewidth=1.5,
    #             )
    # sns.stripplot(x='model', y='vals', data=df,
    #               hue='small_large',
    #               dodge=True,
    #               palette=['dodgerblue', 'orange'],
    #               ax=g,
    #               size=8,
    #               edgecolor='k',
    #               linewidth=1.5,
    #               )
    #
    # plt.show()
    # quit()
    # for ax in g.axes.flat:
    #     for line in ax.lines:
    #         if line.get_linestyle() == '-':
    #             line.set_color('white')
    #             line.set_linewidth(1.5)

def plot_FigureS1_bars_region(region='Occipital', big_voxelwise=True,
                              get_betas=False, no_lines_stars=False,
                              fp=None):
    trial_similarity = 'corr'
    second_order = 'spear'
    RDM_method = 'within_nan'
    stdize_by_run = False
    regress_row = False

    kwargs = {'fp': None,
              # 'fp0': None,
              # 'fp1': None,
              'trial_similarity': trial_similarity,
              'second_order': second_order,
              'RDM_method': RDM_method,
              'stdize_by_run': stdize_by_run,
              'regress_row': regress_row, #'four_tasks': four_tasks,
              }

    if fp is None:
        four_tasks = '7'
        fps = prep_fps(four_tasks)
    else:
        fps = [fp]
    sns = get_sns('all')['healthy']

    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}

    conds = [(False, 'Local'), (False, 'Distributed'),
             (True, 'Local'), (True, 'Distributed')]

    if fp is not None:
        for tup in bad_tups:
            sns = [sn for sn in sns if (sn, fp) != tup]

    # conds = [(False, 'Local'), (False, 'Distributed'),
    #          (True, 'Local'), (True, 'Distributed')]


    # only distributed
    # conds = [('Occipital', False), ('Occipital', True),
    #          ('IT', False), ('IT', True),]

    cond2betas = {}
    for cond in conds:
        cond2betas[cond] = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            # kwargs['cv'] = False
            kwargs['return_dif'] = True
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            if (sn, fp) in bad_tups:
                continue

            for cond in conds:
                # if cond[0] in ['Occipital', 'IT']:
                #     kwargs['semantic'] = cond[1]
                #     if big_voxelwise:
                #         kwargs['ROI_focus'] = f'{cond[0]}_BOLD_ctrl_cmb'
                #     else:
                #         kwargs['ROI_focus'] = f'{cond[0]}_BOLD'
                #
                #     kwargs['ROIs_ctrl'] = []
                # else:
                kwargs['semantic'] = cond[0]
                if cond[1] == 'Local':
                    kwargs['ROI_focus'] = f'{region}_M_{trial_similarity}'
                    if big_voxelwise:
                        kwargs['ROIs_ctrl'] = []
                    else:
                        kwargs['ROIs_ctrl'] = [f'{region}_M_{trial_similarity}']

                        # kwargs['ROIs_ctrl'] = [f'{region}_BOLD_cmb']
                else:
                    if big_voxelwise:
                        kwargs['ROI_focus'] = f'{region}_BOLD_cmb'
                        kwargs['ROIs_ctrl'] = []
                    else:
                        kwargs['ROI_focus'] = f'{region}_BOLD'
                        kwargs['ROIs_ctrl'] = [f'{region}_M_{trial_similarity}']
                        # kwargs['ROI_focus'] = f'{region}_BOLD_cmb'
                    # if big_voxelwise:
                    #     kwargs['ROIs_ctrl'] = []
                    # else:
                    #     kwargs['ROIs_ctrl'] = []
                if kwargs['semantic'] == False:
                    kwargs['semantic'] = (False, 0)

                beta1, test, _ = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                          verbose=-1, easy_override=True)

                cond2betas[cond][i, j] = beta1 * 1000

    cond2betas = {cond: np.nanmean(betas, axis=1) for cond, betas
                  in cond2betas.items()}
    if get_betas:
        return cond2betas
    # plt.figure(figsize=(4.2, 6))
    plt.figure(figsize=(5.3, 7.5))


    cond2Ms = {cond: np.nanmean(betas) for cond, betas in cond2betas.items()}
    cond2SEs = {cond: np.nanstd(betas) / np.sqrt(len(betas))
                for cond, betas in cond2betas.items()}

    itr = (cond2betas[conds[0]] - cond2betas[conds[1]] -
           cond2betas[conds[2]] + cond2betas[conds[3]])
    t, p = stats.ttest_1samp(itr, 0)

    oc_ef = cond2betas[conds[0]] - cond2betas[conds[1]]
    oc_t, oc_p = stats.ttest_1samp(oc_ef, 0)
    ITL_ef = cond2betas[conds[2]] - cond2betas[conds[3]]
    ITL_t, ITL_p = stats.ttest_1samp(ITL_ef, 0)
    print(conds)

    print(f'Interaction: {t=:.3f}, {p=:.3f}')
    print(f'\tOccipital: {oc_t=:.3f}, {oc_p=:.3f}')
    print(f'\tITL: {ITL_t=:.3f}, {ITL_p=:.3f}')

    Ms = np.array([cond2Ms[cond] for cond in conds])
    SEs = np.array([cond2SEs[cond] for cond in conds])
    max_yerr = max(Ms + SEs)
    # height = max_yerr * 1.05

    plt.rcParams.update({'font.sans-serif': 'Arial'})
    colors = ['dodgerblue' if (isinstance(cond[1], bool) or 'Local' in cond[1])
              else ('orange' if big_voxelwise else 'red')
              for cond in conds]
    # print(Ms)
    # quit()
    plt.bar([0, 1, 2, 3], Ms, yerr=SEs, color=colors,
            capsize=5, linewidth=0., edgecolor='k',
            alpha=.75,
            )

    for x in [0, 1, 2, 3]:
        plt.plot([x - .4, x - .4],
                 [0, Ms[x]], color='k', linewidth=0.75, alpha=1.0)
        plt.plot([x + .4, x + .4],
                 [0, Ms[x]], color='k', linewidth=0.75, alpha=1.0)
        plt.plot([x - .4, x + .4],
                 [Ms[x], Ms[x]], color='k', linewidth=0.75, alpha=1.0)

    height, floor, pad = plot_boxen(cond2betas, big_voxelwise)


    plt.xticks([0, 1, 2, 3],
               ['Mean\nsmall', 'Large' if big_voxelwise else 'Large\n(Averages)',
                'Mean\nsmall', 'Large' if big_voxelwise else 'Large\n(Averages)'
                ],
               fontsize=19)

    # plt.text(0.5, 0 - height * .19, 'Occipital',
    #          fontsize=22, ha='center')
    # plt.text(2.5, 0 - height * .19, 'Temporal',
    #          fontsize=22, ha='center')

    # plt.text(0.5, 0 - height * .35, 'Perceptual',
    #          fontsize=22, ha='center')
    # plt.text(2.5, 0 - height * .35, 'Semantic',
    #          fontsize=22, ha='center')
    # max_yerr = np.max(Ms) + (height - floor) * .3

    # lower_signif_line = max_yerr * 1.04
    # upper_signif_line = max_yerr * 1.1

    lower_signif_line = (height - floor + 2*pad) * .88 + floor - pad
    upper_signif_line = (height - floor + 2*pad) * .93 + floor - pad
    shift_down = (height - floor + 2*pad) * .05
    if not no_lines_stars:
        plt.plot([0, 1], [lower_signif_line, lower_signif_line], color='k',
                 linewidth=0.75)
        plt.plot([0.5, 0.5], [lower_signif_line, upper_signif_line], color='k',
                 linewidth=0.75)
        plt.plot([2, 3], [lower_signif_line, lower_signif_line], color='k',
                 linewidth=0.75)
        plt.plot([2.5, 2.5], [lower_signif_line, upper_signif_line], color='k',
                 linewidth=0.75)
        plt.plot([0.5, 2.5], [upper_signif_line, upper_signif_line], color='k',
                 linewidth=0.75)

        # stars_height = height * .92 if stars != 'NS' else height * 1.02
        itr_stars = get_stars(p)
        itr_stars = itr_stars.replace('NS', ' ')
        stars_fs = 28 if itr_stars not in ['NS', '†'] else 24
        # if 'NS' not in itr_stars:
        plt.text(1.5,
                 upper_signif_line - shift_down if itr_stars not in ['NS', '†'] else
                 upper_signif_line - shift_down,
                 itr_stars, ha='center', va='center',
                 fontsize=stars_fs, )

        oc_stars = get_stars(oc_p)
        if 'NS' not in oc_stars:
            stars_fs = 28
            plt.text(0.5, lower_signif_line - shift_down,
                     oc_stars, ha='center', va='center',
                     fontsize=stars_fs, )
        itl_stars = get_stars(ITL_p)
        if 'NS' not in itl_stars:
            stars_fs = 28
            plt.text(2.5, lower_signif_line - shift_down,
                     itl_stars, ha='center', va='center',
                     fontsize=stars_fs, )

    ax = plt.gca()
    if height > 0 and np.max(Ms) > 0:
        ytick_low = (floor // 10 * 10)
        ytick_high = (height // 10 * 10)
        ticks = list(range(int(ytick_low), int(ytick_high) + 1, 10))
        ticks_corr = [tick / 1000 for tick in ticks]
        plt.yticks(ticks, ticks_corr,
                   fontsize=20)

        ax.tick_params(axis='y', which='major', pad=5, )

        for label in ax.get_yticklabels():
            label.set_ha('right')
        # print('test')

        # quit()

        # plt.yticks(fontsize=20)
    # plt.locator_params(axis='y', nbins=6)

    plt.ylabel('Mean correlation' if big_voxelwise else 'Mean beta',
               fontsize=24, labelpad=0)
    # ax.yaxis.set_label_coords(-0.33, 0.5)
    ax.yaxis.set_label_coords(-0.25, 0.5)

    plt.gca().spines[['top', 'right', 'bottom']].set_visible(False)
    # if height > 0 and np.max(Ms) > 0:
    #     plt.ylim(0, height * 1.07)
    plt.ylim(floor - pad, height + pad)
    plt.text(0.5, floor - pad * 12, 'Perceptual',
             fontsize=22, ha='center')
    plt.text(2.5, floor - pad * 12, 'Semantic',
             fontsize=22, ha='center')

    plt.subplots_adjust(bottom=0.26, left=0.29, right=.98, top=.98)

    # plt.text(2.5, 0 - (top_level - bottom) * .225, 'Target After',
    #          fontsize=17.5, ha='center')

    # plt.bar(cond2Ms.keys(), cond2Ms.values(), yerr=cond2SEs.values(),
    #         capsize=5, linewidth=1., edgecolor='k')
    plt.show()

def plot_Figure5_bars():
    do_regr_dif(semantic=False, inter=False, strict_corr=False,
                big_voxelwise=False)
    do_regr_dif(semantic=True, inter=False, strict_corr=False,
                big_voxelwise=False)
    # do_regr_dif(semantic=True, inter=False, strict_corr=False)
    quit()

def plot_FigureS1_bars(big_voxelwise=False):
    # plot_FigureS1_bars_region('Occipital', big_voxelwise=big_voxelwise)
    # plot_FigureS1_bars_region('ITL', big_voxelwise=big_voxelwise)
    # plot_FigureS1_bars_region('Parietal', big_voxelwise=big_voxelwise)
    # plot_FigureS1_bars_region('PFC', big_voxelwise=big_voxelwise)

    for fp in ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']:
        plot_FigureS1_bars_region('ITL', big_voxelwise=big_voxelwise,
                                  fp=fp)



    # plot_FigureS1_bars_region('PFC_no_OFC', big_voxelwise=False)
    # plot_FigureS1_bars_region('OrG', big_voxelwise=False)
    # plot_FigureS1_bars_region('PFC_no_OFC', big_voxelwise=True)
    # plot_FigureS1_bars_region('OrG', big_voxelwise=True)




import sys
sys.setrecursionlimit(10000)

if __name__ == '__main__':
#     from nilearn import image
#
#     fps = [r'C:/test.nii', r'C:/test.nii']
#     image.load_img(fps)
#     quit()

    plot_FigureS1_bars()
    plot_FigureS1_bars(big_voxelwise=True)

    # plot_Figure5_bars()
