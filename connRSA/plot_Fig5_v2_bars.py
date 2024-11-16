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

def plot_Fig5_v2_kw(kw, fps, big_voxelwise, region, std=False):
    kw['return_dif'] = True
    # kw['cv'] = False

    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    betas_local = np.full((len(sns), len(fps)), np.nan)
    betas_dist = np.full((len(sns), len(fps)), np.nan)
    betas_dif = np.full((len(sns), len(fps)), np.nan)
    corrs_local = np.full((len(sns), len(fps)), np.nan)
    corrs_dist = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kw['sn'] = sn
            kw['fp'] = fp`
            kw['ROI_focus'] = f'{region}_M'
            dist_key = f'{region}_BOLD_cmb' if big_voxelwise else f'{region}_BOLD'
            kw['ROIs_ctrl'] = [dist_key]
            beta1, beta2, dif = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                  verbose=-1, easy_override=False)
            betas_local[i, j] = beta1
            betas_dist[i, j] = beta2
            betas_dif[i, j] = dif

            kw['ROIs_ctrl'] = []
            corr_local, _, _ = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                 verbose=-1, easy_override=False)
            corrs_local[i, j] = corr_local

            kw['ROI_focus'] = dist_key
            corr_dist, _, _ = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                verbose=-1, easy_override=False)
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
    M_corrs_dist = np.nanmean(corrs_dist)
    M_betas_local = np.nanmean(betas_local)
    M_betas_dist = np.nanmean(betas_dist)
    M_corr_beta_local = M_corrs_local - M_betas_local
    M_corr_beta_dist = M_corrs_dist - M_betas_dist

    max_height = max(M_corrs_local, M_corrs_dist, M_betas_local, M_betas_dist)

    if len(fps) > 1:
        title = '- All tasks -\n'
    else:
        mapper = {'bl7_fMRI': 'Baseline task', 'obj7_fMRI': 'Encoding task',
                  'con7_fMRI': 'Conceptual retrieval', 'vis7_fMRI': 'Visual retrieval'}
        title = f'- {mapper[fps[0]]} -\n'
    title += 'Semantic' if kw['semantic'] else 'Perceptual'
    title += f', {region}'
    print(f'- {title} -')
    print(f'{M_corrs_local=:.3f}')
    print(f'{M_corrs_dist=:.3f}')
    print(f'{M_betas_local=:.3f}')
    print(f'{M_betas_dist=:.3f}')
    print()

    if big_voxelwise:
        bar_names = ['Small', 'Large\n(voxelwise)']
        corr_colors = ['red', 'orange']
        beta_colors = ['darkred', 'chocolate']
    else:
        bar_names = ['Small', 'Large\n(averages)']
        corr_colors = ['red', 'dodgerblue']
        beta_colors = ['darkred', 'midnightblue']
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(bottom=0.18, left=0.35, right=0.9, top=0.85)

    plt.bar(bar_names,
            [M_corr_beta_local, M_corr_beta_dist],
            bottom=[M_betas_local, M_betas_dist],
            label='2', color=corr_colors,
            linewidth=1., edgecolor='k')
    plt.bar(bar_names,
            [M_betas_local, M_betas_dist],
            bottom=[0, 0],
            label='2', color=beta_colors,
            linewidth=1., edgecolor='k')
    if M_corr_beta_dist < 0:
        plt.plot([0.6, 1.4], [M_corrs_dist, M_corrs_dist], color='k',
                 linestyle='-', linewidth=1)
        plt.plot([0.6, 1.4], [M_corrs_dist, M_corrs_dist],
                 color='yellow' if big_voxelwise else 'dodgerblue',
                 linestyle='--', linewidth=3)
    elif M_corr_beta_local < 0:
        plt.plot([-0.4, 0.4], [M_corrs_local, M_corrs_local], color='k',
                 linestyle='-', linewidth=1)
        plt.plot([-0.4, 0.4], [M_corrs_local, M_corrs_local],
                 color='red',
                 linestyle='--', linewidth=3)


    # plt.ylim(0, max_height * 1.05)
    if std:
        pass
    else:
        plt.ylim(0, 0.025)

    plt.title(title, fontsize=20)
    plt.ylabel('Mean (beta | correlation)')
    plt.show()
    return

    # print(M_corrs_local)
    # quit()


    t_corrs_local, p_corrs_local = stats.ttest_1samp(
        np.nanmean(corrs_local, axis=1), 0)
    t_dist_local, p_dist_local = stats.ttest_1samp(
        np.nanmean(corrs_dist, axis=1), 0)
    t_betas_local, p_betas_local = stats.ttest_1samp(
        np.nanmean(betas_local, axis=1), 0)
    t_betas_dist, p_betas_dist = stats.ttest_1samp(
        np.nanmean(betas_dist, axis=1), 0)



def plot_Fig5_v2_bars_all(semantic=False, big_voxelwise=True):
    target_ROIs = ['Occipital', 'ITL']
    # target_ROIs = ['Occipital']
    target_ROIs = ['ITL']

    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    fps = ['obj7_fMRI']

    for target_ROI in target_ROIs:
        # regions = set(prep_networks(
        #     network_setting=ROI2NETWORK[target_ROI])[target_ROI])

        # ROIs_match = [ROI for region in regions
        #                   for ROI in get_BNA_ROIs() if region in ROI]
        # ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]

        kwargs = {'semantic': semantic, 'fp': None,
                  'trial_similarity': 'corr',
                  'second_order': 'spear', 'RDM_method': 'within_nan',
                  'stdize_by_run': False, 'regress_row': False,
                  }
        plot_Fig5_v2_kw(kwargs, fps, big_voxelwise, target_ROI)
        # return



if __name__ == '__main__':
    # plot_Fig5_v2_bars_all()
    # plot_Fig5_v2_bars_all(big_voxelwise=False, semantic=True)
    # plot_Fig5_v2_bars_all(semantic=True,)
    plot_Fig5_v2_bars_all(semantic=False, big_voxelwise=False)

