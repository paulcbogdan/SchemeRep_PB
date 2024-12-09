import numpy as np

from Utils.atlas_funcs import get_BNA_ROIs
from Utils.pickle_wrap_funcs import pickle_wrap
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.conn_regress import do_regr_RSA_sn
from connRSA.single_trial_conn import prep_fps
from networks.old.networks import prep_networks
from org_sns import get_sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats

from organize_bhv import sort_df_sn


def run_regress(region='Occipital', big_voxelwise=True,):
    conds = [(False, 0), True]
    sp2betas = {}
    sp2corrs = {}

    fps = prep_fps('7')
    # fps = ['bl7_fMRI', 'obj7_fMRI',
    #        'con7_fMRI', 'vis7_fMRI']

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}

    for sp in conds:
        sp2betas[sp] = np.full((len(sns), len(fps)), np.nan)
        sp2corrs[sp] = np.full((len(sns), len(fps)), np.nan)

    kwargs = {'fp': None,
              'trial_similarity': 'corr',
              'second_order': 'spear',
              'RDM_method': 'within_nan',
              'stdize_by_run': False,
              'regress_row': False,
              }


    for sp in conds:
        kwargs['semantic'] = sp
        for i, sn in enumerate(tqdm(sns, desc='running sns')):
            for j, fp in enumerate(fps):
                if (sn, fp) in bad_tups:
                    continue
                kwargs['return_dif'] = True
                kwargs['sn'] = sn
                kwargs['fp'] = fp
                if big_voxelwise:
                    kwargs['ROI_focus'] = f'{region}_BOLD_cmb'
                else:
                    kwargs['ROI_focus'] = f'{region}_BOLD'

                regions = set(prep_networks(
                    network_setting=ROI2NETWORK[region])[region])
                ROIs_match = [ROI for region in regions
                              for ROI in get_BNA_ROIs() if region in ROI]
                ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]
                ROI_lvl_control = sorted(ROI_lvl_control)

                kwargs['ROIs_ctrl'] = ROI_lvl_control
                beta1, _, _ = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                          verbose=-1, easy_override=False)
                sp2betas[sp][i, j] = beta1
                kwargs['ROIs_ctrl'] = []
                corr1, _, _ = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                          verbose=-1, easy_override=False)
                sp2corrs[sp][i, j] = corr1

    cond1_vals = np.nanmean(sp2betas[conds[0]], axis=1)
    cond2_vals = np.nanmean(sp2betas[conds[1]], axis=1)

    height = np.quantile([cond1_vals, cond1_vals], .98)
    floor = np.quantile([cond1_vals, cond1_vals], .02)
    sns_out1 = (cond1_vals > height) | (cond1_vals < floor)
    sns_out2 = (cond2_vals > height) | (cond2_vals < floor)
    sns_out = sns_out1 | sns_out2


    t1, p1 = stats.ttest_1samp(cond1_vals, 0)
    t2, p2 = stats.ttest_1samp(cond2_vals, 0)
    t12, p12 = stats.ttest_rel(cond1_vals, cond2_vals)
    print(f'Perceptual: {t1=:.3f}, {p1=:.4f}')
    print(f'Semantic: {t2=:.3f}, {p2=:.4f}')
    print(f'Perceptual vs. Semantic: {t12=:.3f}, {p12=:.4f}')

    for sp, betas in sp2betas.items():
        sp2betas[sp] = np.nanmean(betas, axis=1)
        sp2betas[sp] = sp2betas[sp][~sns_out]
        # print(sns_out)
        # quit()

    cond2Ms = {cond: np.nanmean(betas) for cond, betas in sp2betas.items()}
    cond2SEs = {cond: np.nanstd(betas) / np.sqrt(len(betas))
                for cond, betas in sp2betas.items()}

    Ms = np.array([cond2Ms[cond] for cond in conds])
    SEs = np.array([cond2SEs[cond] for cond in conds])
    t = Ms/SEs

    Ms_corr = np.array([np.nanmean(corr) for corr in sp2corrs.values()])
    SEs_corr = {cond: np.nanstd(betas) / np.sqrt(len(betas))
                for cond, betas in sp2corrs.items()}
    SEs_corr = np.array([SEs_corr[cond] for cond in conds])

    ts_corr = Ms_corr / SEs_corr

    # print(f'{Ms=}')
    # print(f'{t=}')
    # print(f'{Ms_corr=}')
    # print(f'{ts_corr=}')
    # quit()


    Ms_bottom = np.copy(Ms)
    Ms_bottom[Ms_bottom < 0] = 0
    M_corr_beta = Ms_corr - Ms_bottom

    # fig = plt.figure(figsize=(4.2, 5))
    fig = plt.figure(figsize=(5.3, 4))
    # plt.rcParams.update({'font.size': 20,
    #                      'font.sans-serif': 'Arial'})

    # plt.ylim(floor, height)

    color = 'chocolate' if big_voxelwise else 'darkred'
    color_corr = 'orange' if big_voxelwise else 'red'

    # plt.bar([0, 1], Ms_corr, color=color_corr,
    #         capsize=5, linewidth=0,
    #         edgecolor='k', width=0.7)

    width = .5
    plt.bar([0, 1], Ms, #yerr=SEs,
            color=color,
            capsize=5, linewidth=0,
            edgecolor='k', width=width,
            alpha=.75)

    plt.bar([0, 1], M_corr_beta,
            color=color_corr, bottom=Ms_bottom,
            capsize=5, linewidth=0,
            edgecolor='k', width=width,
            alpha=.75)

    plt.rcParams.update({'font.sans-serif': 'Arial'})

    if region in ['IT', 'ITL']:
        plt.text(1, -0.0004, '  ***  ', fontsize=28, ha='center',
                 color='w', va='center')

    lw = 1.5
    for x in [0, 1]:
        if region not in ['IT', 'ITL'] or x == 0:
            plt.plot([x, x], [Ms[x] - SEs[x], Ms[x] + SEs[x]], color='k',
                     linewidth=lw, alpha=1.0)
            plt.plot([x - 0.05, x + 0.05], [Ms[x] - SEs[x], Ms[x] - SEs[x]], color='k',
                     linewidth=lw, alpha=1.0)
            plt.plot([x - 0.05, x + 0.05], [Ms[x] + SEs[x], Ms[x] + SEs[x]], color='k',
                     linewidth=lw, alpha=1.0)
        else:
            plt.plot([x, x], [Ms[x] - SEs[x], Ms[x] + SEs[x]], color='w',
                     linewidth=lw, alpha=1.0)
            plt.plot([x - 0.05, x + 0.05], [Ms[x] - SEs[x], Ms[x] - SEs[x]], color='w',
                     linewidth=lw, alpha=1.0)
            plt.plot([x - 0.05, x + 0.05], [Ms[x] + SEs[x], Ms[x] + SEs[x]], color='w',
                     linewidth=lw, alpha=1.0)

        if M_corr_beta[x] > 0:
            plt.plot([x - width / 2, x - width / 2],
                     [0, Ms_corr[x]], color='k', linewidth=lw, alpha=1.0)
            plt.plot([x + width / 2, x + width / 2],
                     [0, Ms_corr[x]], color='k', linewidth=lw, alpha=1.0)
            plt.plot([x - width / 2, x + width / 2],
                     [Ms_corr[x], Ms_corr[x]], color='k', linewidth=lw, alpha=1.0)
            if Ms[x] > 0:
                plt.plot([x - width / 2, x + width / 2],
                         [Ms[x], Ms[x]], color='w', linewidth=lw, alpha=1.0)

            else:
                plt.plot([x - width / 2, x + width / 2],
                         [Ms[x], Ms[x]], color='k', linewidth=lw, alpha=1.0)
        else:
            plt.plot([x - width / 2, x - width / 2],
                     [0, Ms[x]], color='k', linewidth=lw, alpha=1.0)
            plt.plot([x + width / 2, x + width / 2],
                     [0, Ms[x]], color='k', linewidth=lw, alpha=1.0)



    plt.plot([-.5, 1.5], [0, 0], linewidth=0.5, color='k')
    plt.xlim(-.5, 1.5)

    for spot, (cond, betas) in enumerate(sp2betas.items()):
        # betas = Ms_corr[cond]
        # continue
        # v = np.nanmean(betas, axis=1)
        v = betas
        jitter = np.random.uniform(-0.13, 0.13, len(v))
        dist = np.abs((v - np.mean(v)) / np.std(v))
        dist = np.clip(dist, 0, 3)
        dist = (3 - dist) / 3
        jitter *= dist
        plt.scatter(np.full(len(v), spot) + jitter, v,
                    color=color, s=25, alpha=0.25,
                    linewidth=1,
                    )
    ax = plt.gca()
    ax.tick_params(axis='y', which='major', pad=5, )
    plt.ylabel('Mean beta & corr.', fontsize=24, labelpad=10)
    plt.yticks(fontsize=20)

    plt.gca().spines[['top', 'right', 'bottom']].set_visible(False)
    plt.xticks([0, 1], ['Perceptual', 'Semantic'])


    plt.subplots_adjust(bottom=0.26, left=0.29, right=.98, top=.98)

    # plt.subplots_adjust(left=0.4, right=.9)

    plt.savefig(f'result_pics/test_{region}.png', dpi=300)

    plt.show()

# from matplotlib.text import Text
# ax = plt.gca()
# def custom_ylabel(ax, text):
#     # Split the text into words
#     words = text.split()
#
#     # Create custom Text objects for each colored word
#     dark_red_text = Text(0, 0, words[0], color=(0.5, 0, 0, 0.7),
#                          rotation=90, verticalalignment='center')
#     red_text = Text(0, 0, ' ' + ' '.join(words[1:]), color=(1, 0, 0, 0.7),
#                     rotation=90, verticalalignment='center')
#
#     # Remove the existing ylabel
#     ax.set_ylabel('')
#
#     # Add custom text
#     ax.text(-0.1, 0.5, dark_red_text._text,
#             color=dark_red_text.get_color(),
#             transform=ax.transAxes,
#             rotation=90,
#             verticalalignment='center')
#     ax.text(-0.1, 0.5, red_text._text,
#             color=red_text.get_color(),
#             transform=ax.transAxes,
#             rotation=90,
#             verticalalignment='center')
#
#
# # Example plot
# plt.plot([1, 2, 3], [1, 2, 3])
#
# # Apply custom y-label
# custom_ylabel(ax, 'Dark Red red Words')
#
# plt.tight_layout()
# plt.show()


if __name__ == '__main__':
    # ax = plt.gca()
    # ax.set_ylabel('Dark Red Word red Word',
    #               color={'Dark Red': (0.5, 0, 0, 0.7),
    #                      'red': (1, 0, 0, 0.7)},
    #               fontsize=12)
    # plt.show()
    # quit()

    BIG_VOXELWISE = False
    # run_regress('Occipital', big_voxelwise=BIG_VOXELWISE)
    run_regress('ITL', big_voxelwise=BIG_VOXELWISE)
    # run_regress('Parietal', big_voxelwise=BIG_VOXELWISE)
    # run_regress('PFC', big_voxelwise=BIG_VOXELWISE)