
from Utils.atlas_funcs import get_BNA_ROIs
import utils
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.conn_regress import do_regr_RSA_sn
from connRSA.single_trial_conn import prep_fps
from networks.old.networks import prep_networks
from org_sns import get_sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# TODO: box bar

def plot_Fig5_v2_kw(kw, fps, big_voxelwise, region, std=False,
                    easy_override=False, ax=None, plot_i=0):
    kw['return_dif'] = True
    # kw['cv'] = False

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']

    bad_sns_euc = ['132', '138', '224', '234']
    if kw['trial_similarity'] == 'euc':
        sns = [sn for sn in sns if sn not in bad_sns_euc]

    sns = [sn for sn in sns if sn not in bad_sns]
    betas_local = np.full((len(sns), len(fps)), np.nan)
    betas_dist = np.full((len(sns), len(fps)), np.nan)
    betas_dif = np.full((len(sns), len(fps)), np.nan)
    corrs_local = np.full((len(sns), len(fps)), np.nan)
    corrs_dist = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        # if sn in ['132', '138', '224', '234']: continue
        for j, fp in enumerate(fps):
            # print(f'{sn}, {fp}')
            kw['sn'] = sn
            kw['fp'] = fp
            kw['ROI_focus'] = f'{region}_M_corr'
            dist_key = f'{region}_BOLD_cmb' if big_voxelwise else f'{region}_BOLD'
            # dist_key = f'{region}_BOLD_ctrl_cmb'
            kw['ROIs_ctrl'] = [dist_key]
            kw['return_dif'] = True
            # easy_override = True
            beta1, beta2, dif = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                  verbose=-1,
                                                  easy_override=easy_override)
            # print(beta1)
            # print(beta2)
            betas_local[i, j] = beta1
            betas_dist[i, j] = beta2
            betas_dif[i, j] = dif

            kw['ROIs_ctrl'] = []
            # print(kw)
            corr_local, _, _ = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                 verbose=-1, easy_override=easy_override)
            # print(corr_local)
            # quit()
            corrs_local[i, j] = corr_local

            kw['ROI_focus'] = dist_key

            corr_dist, _, _ = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                verbose=-1, easy_override=easy_override)
            corrs_dist[i, j] = corr_dist

    assert np.sum(np.isnan(corrs_local)) == 0
    assert np.sum(np.isnan(corrs_dist)) == 0
    assert np.sum(np.isnan(betas_local)) == 0
    assert np.sum(np.isnan(betas_dist)) == 0

    corrs_local = np.nanmean(corrs_local, axis=1)
    corrs_dist = np.nanmean(corrs_dist, axis=1)
    betas_local = np.nanmean(betas_local, axis=1)
    betas_dist = np.nanmean(betas_dist, axis=1)

    if std:
        corrs_local /= np.std(corrs_local)
        corrs_dist /= np.std(corrs_dist)
        betas_local /= np.std(betas_local)
        betas_dist /= np.std(betas_dist)

    M_corrs_local = np.nanmean(corrs_local)
    SD_corrs_local = np.nanstd(corrs_local) / np.sqrt(len(corrs_local))
    M_corrs_dist = np.nanmean(corrs_dist)
    SD_corrs_dist = np.nanstd(corrs_dist) / np.sqrt(len(corrs_local))
    M_betas_local = np.nanmean(betas_local)
    SD_betas_local = np.nanstd(betas_local) / np.sqrt(len(corrs_local))
    M_betas_dist = np.nanmean(betas_dist)
    SD_betas_dist = np.nanstd(betas_dist) / np.sqrt(len(corrs_local))
    M_corr_beta_local = M_corrs_local - M_betas_local
    M_corr_beta_dist = M_corrs_dist - M_betas_dist

    t_corr, p_corr = stats.ttest_rel(corrs_local, corrs_dist)
    print(f'{t_corr=:.3f}, {p_corr=:.3f}')
    t_beta, p_beta = stats.ttest_rel(betas_local, betas_dist)
    print(f'{t_beta=:.3f}, {p_beta=:.3f}')

    max_height = max(M_corrs_local, M_corrs_dist, M_betas_local, M_betas_dist)

    if len(fps) > 1:
        title = 'All tasks: '
    else:
        mapper = {'bl7_fMRI': 'Baseline task', 'obj7_fMRI': 'Encoding task',
                  'con7_fMRI': 'Conceptual retrieval', 'vis7_fMRI': 'Visual retrieval'}
        title = f'{mapper[fps[0]]}:\n'
    if isinstance(kw['semantic'], bool) and kw['semantic']:
        title += 'Semantic'
    elif isinstance(kw['semantic'], bool) and not kw['semantic']:
        title += 'DNN layer 2'
    elif isinstance(kw['semantic'], tuple) and kw['semantic'][0] != 'llama':
        assert not kw['semantic'][0]
        title += f'DNN layer {kw["semantic"][1]}'
    elif isinstance(kw['semantic'], tuple) and kw['semantic'][0] == 'llama':
        title += f' LLAMA: {kw["semantic"][1]}/{kw["semantic"][2]}/{kw["semantic"][3]}'
    title += f'\n{region}'
    print(f'- {title} -')
    # print(f'{M_corrs_local=:.3f}')
    # print(f'{M_corrs_dist=:.3f}')
    # print(f'{M_betas_local=:.3f}')
    # print(f'{M_betas_dist=:.3f}')
    # print()

    if big_voxelwise:
        bar_names = ['Small', 'Large\n(voxelwise)']
        corr_colors = ['dodgerblue', 'orange']
        beta_colors = ['midnightblue', 'chocolate']
    else:
        bar_names = ['Small', 'Large\n(averages)']
        corr_colors = ['dodgerblue', 'red']
        beta_colors = ['midnightblue', 'darkred']
    if ax is not None:
        plt.sca(ax)
    else:
        fig = plt.figure(figsize=(5, 5))
        fig.subplots_adjust(bottom=0.18, left=0.35, right=0.9, top=0.85)

    plt.bar(bar_names,
            [M_corr_beta_local, M_corr_beta_dist],
            bottom=[M_betas_local, M_betas_dist],
            yerr=[SD_corrs_local, SD_corrs_dist],
            error_kw={'ecolor': 'k', 'capsize': 6,  'linewidth': 2},
            label='2', color=corr_colors,
            linewidth=1., edgecolor='k')
    # plt.errorbar([-0.05, 0.95],
    #             [M_corrs_local, M_corrs_dist],
    #             yerr=[SD_corrs_local, SD_corrs_dist],
    #             linewidth=0, elinewidth=2, color='k',
    #             capsize=6, marker='o')
    plt.bar(bar_names,
            [M_betas_local, M_betas_dist],
            bottom=[0, 0],
            # yerr=[SD_betas_local, SD_betas_dist],
            # error_kw={'ecolor': 'w', 'capsize': 6,  'linewidth': 2},
            label='2', color=beta_colors,
            linewidth=1., edgecolor='k')
    # plt.errorbar([0.05, 1.05],
    #             [M_betas_local, M_betas_dist],
    #             yerr=[SD_betas_local, SD_betas_dist],
    #             linewidth=0, elinewidth=2, color='k',
    #             capsize=6, marker='o')
    # plt.errorbar([0.1, 1.1],
    #             [M_betas_local, M_betas_dist],
    #             yerr=[SD_betas_local, SD_betas_dist],
    #             linewidth=0, elinewidth=1, color='k',
    #             capsize=6, marker='o')
    if M_corr_beta_dist < 0:
        plt.plot([0.6, 1.4], [M_corrs_dist, M_corrs_dist], color='k',
                 linestyle='-', linewidth=1)
        plt.plot([0.6, 1.4], [M_corrs_dist, M_corrs_dist],
                 color='yellow' if big_voxelwise else 'red',
                 linestyle='--', linewidth=3)
    elif M_corr_beta_local < 0:
        plt.plot([-0.4, 0.4], [M_corrs_local, M_corrs_local], color='k',
                 linestyle='-', linewidth=1)
        plt.plot([-0.4, 0.4], [M_corrs_local, M_corrs_local],
                 color='dodgerblue',
                 linestyle='--', linewidth=3)


    plt.ylim(0, max_height * 1.1)
    # plt.ylim(0, 0.02)
    # if std:
    #     pass
    # else:
    #     plt.ylim(0, 0.055)
    #     plt.yticks([0, .01, .02, .03, .04, .05])

    # plt.ylim(0, None)
    plt.title(title, fontsize=20)
    if plot_i == 0:
        plt.ylabel('Mean (beta | correlation)')
    # plt.gcf().subplots_adjust(bottom=0.18, left=0.15, right=0.95, top=0.85, wspace=0.2)

    # plt.show()
    return

    # print(M_corrs_local)
    # quit()
    # t_corrs_local, p_corrs_local = stats.ttest_1samp(
    #     np.nanmean(corrs_local, axis=1), 0)
    # t_dist_local, p_dist_local = stats.ttest_1samp(
    #     np.nanmean(corrs_dist, axis=1), 0)
    # t_betas_local, p_betas_local = stats.ttest_1samp(
    #     np.nanmean(betas_local, axis=1), 0)
    # t_betas_dist, p_betas_dist = stats.ttest_1samp(
    #     np.nanmean(betas_dist, axis=1), 0)



def plot_Fig5_v2_bars_all(big_voxelwise=True):
    target_ROIs = ['Occipital', 'ITL', 'Parietal', 'PFC']

    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI',  'vis7_fMRI' ]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI',  'vis7_fMRI' ]
    # fps = ['obj7_fMRI']

    semantic_tup0 = ('llama', 'input', 0, 'obj')
    semantic_l = [semantic_tup0]

    all_llama_cats = ['gate_proj_in', 'up_proj_in', 'down_proj_in', 'act_fn_in',
                      'gate_proj_out', 'up_proj_out', 'down_proj_out', 'act_fn_out',
                      'q_proj', 'k_proj', 'v_proj', 'attn_weights', 'attn_output',
                      'input']
    all_llama_layers = list(range(16))

    for llama_layer in all_llama_layers:
        for llama_cat in all_llama_cats:
        # for llama_layer in all_llama_layers:
            semantic_l.append(('llama', llama_cat, llama_layer, 'obj'))
            semantic_l.append(('llama', llama_cat, llama_layer, 'scn'))


    # semantic_l = semantic_l[::-1]
    # for DNN_layer in DNN_layers:
    for semantic in semantic_l:
        print(f'{semantic=}')

        # semantic = ('llama', 'input', 1, 'obj')
        # semantic = ('llama', 'gate_proj_in', 0, 'obj')

        plt.rcParams.update({'font.size': 20})
        fig, axs = plt.subplots(1, len(target_ROIs), figsize=(5 * len(target_ROIs), 5))
        # fig.subplots_adjust(bottom=0.18, left=0.35, right=0.9, top=0.85)
        fig.subplots_adjust(bottom=0.18, left=0.15, right=0.95, top=0.85, wspace=0.3)

        for i, target_ROI in enumerate(target_ROIs):
            # regions = set(prep_networks(
            #     network_setting=ROI2NETWORK[target_ROI])[target_ROI])

            # ROIs_match = [ROI for region in regions
            #                   for ROI in get_BNA_ROIs() if region in ROI]
            # ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]

            kwargs = {'semantic': semantic,
                      'fp': None,
                      'trial_similarity': 'corr',
                      'second_order': 'spear', 'RDM_method': 'within_nan',
                      'stdize_by_run': False,
                      'regress_row': False,
                      }
            plot_Fig5_v2_kw(kwargs, fps, big_voxelwise, target_ROI,
                            ax=axs[i] , plot_i=i)
        plt.show()
    quit()



if __name__ == '__main__':
    # plot_Fig5_v2_bars_all()
    # plot_Fig5_v2_bars_all(big_voxelwise=False, semantic=True)
    # plot_Fig5_v2_bars_all(semantic=True,) 
    plot_Fig5_v2_bars_all(big_voxelwise=False)

