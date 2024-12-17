from collections import defaultdict

from Utils.atlas_funcs import get_BNA_ROIs
import utils
from Utils.pickle_wrap_funcs import pickle_wrap
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.conn_regress import do_regr_RSA_sn
from connRSA.single_trial_conn import prep_fps
from llama.color_test import print_colored_list
from networks.old.networks import prep_networks
from org_sns import get_sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import cache

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
    for i, sn in enumerate(tqdm(sns, desc='looping through subjects')):
        for j, fp in enumerate(fps):
            # print(f'{sn} | {fp}')
            kw['sn'] = sn
            kw['fp'] = fp
            kw['ROI_focus'] = f'{region}_M_corr'
            dist_key = f'{region}_BOLD_cmb' if big_voxelwise else f'{region}_BOLD'
            kw['ROIs_ctrl'] = [dist_key]
            kw['return_dif'] = True
            beta1, beta2, dif = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                  verbose=-1,
                                                  easy_override=easy_override,
                                                  dir_branches=100)
            betas_local[i, j] = beta1
            betas_dist[i, j] = beta2
            betas_dif[i, j] = dif

            kw['ROIs_ctrl'] = []
            corr_local, _, _ = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                 verbose=-1, easy_override=easy_override,
                                                 dir_branches=100)

            corrs_local[i, j] = corr_local
            kw['ROI_focus'] = dist_key
            corr_dist, _, _ = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                verbose=-1, easy_override=easy_override,
                                                dir_branches=100)
            corrs_dist[i, j] = corr_dist

    assert np.sum(np.isnan(corrs_local)) == 0
    assert np.sum(np.isnan(corrs_dist)) == 0
    assert np.sum(np.isnan(betas_local)) == 0
    assert np.sum(np.isnan(betas_dist)) == 0

    corrs_local *= 1_000
    corrs_dist *= 1_000
    betas_local *= 1_000
    betas_dist *= 1_000

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



    max_height = max(M_corrs_local, M_corrs_dist, M_betas_local, M_betas_dist)

    if len(fps) > 1:
        title = 'All tasks: '
    else:
        mapper = {'bl7_fMRI': 'Baseline task', 'obj7_fMRI': 'Encoding task',
                  'con7_fMRI': 'Conceptual retrieval', 'vis7_fMRI': 'Visual retrieval'}
        title = f'{mapper[fps[0]]}:\n'
    title = ''
    if isinstance(kw['semantic'], bool) and kw['semantic']:
        title += 'Semantic'
    elif isinstance(kw['semantic'], bool) and not kw['semantic']:
        title += 'DNN layer 2'
    elif isinstance(kw['semantic'], tuple) and kw['semantic'][0] != 'llama' and kw['semantic'][0] != 'BERT':
        assert not kw['semantic'][0]
        title += f'DNN layer {kw["semantic"][1]}'
    elif isinstance(kw['semantic'], tuple) and kw['semantic'][0] == 'llama':
        title += f'{kw["semantic"][1]}/{kw["semantic"][2]}/{kw["semantic"][3]}'
    elif isinstance(kw['semantic'], tuple) and kw['semantic'][0] == 'BERT':
        title += f'BERT layer {kw["semantic"][2]}'
    elif isinstance(kw['semantic'], list):
        title += ' super list'
    title += f'\n{region}'
    print()
    print(f'- {title.replace("\n", " ")} -')
    t_corr, p_corr = stats.ttest_rel(corrs_local, corrs_dist)
    print(f'Local vs. distributed, corr, t = {t_corr:.2f}, p = {p_corr:.3f}')
    t_local_corr_1samp, p_local_corr_1samp = stats.ttest_1samp(corrs_local, 0)
    print(f'\tLocal 1-samp, corr: t = {t_local_corr_1samp:.2f}, p = {p_local_corr_1samp:.3f}')
    t_dist_corr_1samp, p_dist_corr_1samp = stats.ttest_1samp(corrs_dist, 0)
    print(f'\tDist 1-samp, corr: t = {t_dist_corr_1samp:.2f}, p = {p_dist_corr_1samp:.3f}')

    t_beta, p_beta = stats.ttest_rel(betas_local, betas_dist)
    print(f'Local vs. distributed, beta, t = {t_beta:.2f}, p = {p_beta:.3f}')
    t_local_beta_1samp, p_local_beta_1samp = stats.ttest_1samp(betas_local, 0)
    print(f'\tLocal 1-samp, beta: t = {t_local_beta_1samp:.2f}, p = {p_local_beta_1samp:.3f}')
    t_dist_beta_1samp, p_dist_beta_1samp = stats.ttest_1samp(betas_dist, 0)
    print(f'\tDist 1-samp, beta: t = {t_dist_beta_1samp:.2f}, p = {p_dist_beta_1samp:.3f}')
    return (t_corr, t_local_corr_1samp, t_dist_corr_1samp,
            t_beta, t_local_beta_1samp, t_dist_beta_1samp)

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

    plt.bar(bar_names,
            [M_betas_local, M_betas_dist],
            bottom=[0, 0],
            # yerr=[SD_betas_local, SD_betas_dist],
            # error_kw={'ecolor': 'w', 'capsize': 6,  'linewidth': 2},
            label='2', color=beta_colors,
            linewidth=1., edgecolor='k')
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
    return


@cache
def get_explore_llama(activation_model='meta-llama/Llama-3.3-70b-Instruct'):
    # Redundant: gate_proj_in & up_proj_in
    # Redundant: act_fn_in & gate_proj_out
    all_llama_cats = ['gate_proj_in', 'up_proj_in', 'down_proj_in', 'act_fn_in',
                      'gate_proj_out', 'up_proj_out', 'down_proj_out', 'act_fn_out',
                      'q_proj', 'k_proj', 'v_proj', 'attn_weights', 'attn_output',
                      'input']
    all_llama_cats = ['attn_weights']
    all_llama_cats = ['gate_proj_in']

    # all_llama_layers = list(range(28, 32))

    # all_llama_layers = [4, 5, 6, 7]

    if activation_model == 'meta-llama/Llama-3.2-3b':
        all_llama_layers = list(range(0, 28))
    else:
        all_llama_layers = list(range(0, 80))

    # all_llama_layers = list(range(0, 10))

    normalize = True

    semantic_l = []
    for llama_layer in all_llama_layers:
        for llama_cat in all_llama_cats:
            if llama_cat in ['attn_weights', 'attn_output']:
                semantic_l.append(('llama', llama_cat, llama_layer, 'obj', activation_model,
                                   normalize))
            semantic_l.append(('llama', llama_cat, llama_layer, 'scn', activation_model,
                               normalize))
    return semantic_l

def get_explore_BERT(bert_type='BERT'):
    layers = list(range(0, 13))
    semantic_l = []
    normalize = True
    for layer in layers:
        semantic = ('BERT', bert_type, layer, 'obj', None, normalize)
        # semantic = ('BERT', 'simCSE', layer, 'obj', None, normalize)

        semantic_l.append(semantic)
    return semantic_l

def plot_Fig5_v2_bars_all(big_voxelwise=True):
    target_ROIs = ['Occipital', 'ITL', 'Parietal', 'PFC']

    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI',  'vis7_fMRI' ]
    # fps = ['obj7_fMRI']

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    # semantic_l = [True]
    # semantic_l = semantic_l[3::4]
    # semantic_l = semantic_l[1::2]
    ALL_RESULTS = defaultdict(list)

    target_roi2baddies = defaultdict(list)
    semantic_l = get_explore_BERT()
    semantic_l = [True]
    # semantic_l = get_explore_llama()

    for semantic in semantic_l:

        print(f'\n{semantic=}')
        plt.rcParams.update({'font.size': 20})
        fig.subplots_adjust(bottom=0.18, left=0.15, right=0.95, top=0.85, wspace=0.3)

        if isinstance(semantic, bool) and semantic:
            llama_cat, llama_layer, obj_scn = None, None, None
        else:
            llama_cat = semantic[1]
            llama_layer = semantic[2]
            obj_scn = semantic[3]
        for i, target_ROI in enumerate(target_ROIs):
            kwargs = {'semantic': semantic,
                      'fp': None,
                      'trial_similarity': 'corr',
                      'second_order': 'spear',
                      'RDM_method': 'within_nan',
                      'stdize_by_run': False,
                      'regress_row': False,
                      }
            i_ = i // 2
            j = i % 2

            kw = {'kw': kwargs, 'fps': fps, 'big_voxelwise': big_voxelwise,
                  'region': target_ROI, 'std': False,
                  'easy_override': False, 'ax': axs[i_][j], 'plot_i': i}

            (t_corr, t_local_corr_1samp, t_dist_corr_1samp,
             t_beta, t_local_beta_1samp, t_dist_beta_1samp) = (
                pickle_wrap(plot_Fig5_v2_kw, kwargs=kw,
                            verbose=-1, easy_override=False))

            if t_dist_corr_1samp < 2:
                target_roi2baddies[target_ROI].append(
                    semantic)
            print(f'{target_ROI}: {t_dist_corr_1samp=:.2f}')
            ALL_RESULTS[(target_ROI, llama_cat, obj_scn)].append(t_dist_corr_1samp)
        print_all_results(ALL_RESULTS)
        if len(axs[0][0].lines):
            plt.tight_layout()
            plt.show()
            fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    target_roi2baddies = dict(target_roi2baddies)
    print(f'{target_roi2baddies=}')
    plot_all_results(ALL_RESULTS)
    quit()

def plot_all_results(all_results):
    # print(all_results)
    # quit()
    nested_dict = flat_dict_to_nested(all_results)

    fig, axs = plt.subplots(len(nested_dict), 1, figsize=(10, 5 * len(nested_dict)))
    for j, (region, b_dict) in enumerate(nested_dict.items()):
        print(f'Region: {region}')
        num_types = len(b_dict)
        plt.sca(axs[j])
        plt.title(region)
        # print(list(b_dict))
        # quit()
        for i, (module_type, c_dict) in enumerate(b_dict.items()):
            if 'scn' not in c_dict and 'BERT' not in module_type: continue
            if 'scn' in c_dict:
                values = c_dict['scn']
            elif 'BERT' in module_type:
                values = c_dict['obj']
            layer_nums = list(range(len(values)))
            plt.plot(layer_nums, values, label=module_type if j == 0 else None)
        plt.ylim(0, 10)
        plt.ylabel('t-value')
    # plt.legend(ncol=3)
    fig.legend(loc='lower center',  ncol=3)
    plt.tight_layout()
    plt.show()

    # plt.suptitle(region)



def flat_dict_to_nested(flat_dict):
    nested_dict = {}
    for (a, b, c), value in flat_dict.items():
        # Create nested dictionaries if they don't exist
        if a not in nested_dict:
            nested_dict[a] = {}

        if b not in nested_dict[a]:
            nested_dict[a][b] = {}

        nested_dict[a][b][c] = value
    return nested_dict

def print_all_results(all_results):
    nested_dict = flat_dict_to_nested(all_results)

    for a, b_dict in nested_dict.items():
        print(f'Region: {a}')
        header = ''.join(f'{x:^5}|' for x in range(15))
        print(f'\t{header}')
        for b, c_dict in b_dict.items():
            for c, value in c_dict.items():
                M_v = np.nanmean(value)
                SE_v = np.nanstd(value) / np.sqrt(len(value))
                v_ = [float(f'{v:.2f}') for v in value]
                v_ = [np.max([0, v]) for v in v_]
                v_str = print_colored_list(v_, vmin=0, vmax=10)
                print(f'\t{v_str} | {b}, {c}: M = {M_v:.1f} [{SE_v:.2f}]. ')

if __name__ == '__main__':
    # plot_Fig5_v2_bars_all()
    # plot_Fig5_v2_bars_all(big_voxelwise=False, semantic=True)
    # plot_Fig5_v2_bars_all(semantic=True,) 
    plot_Fig5_v2_bars_all(big_voxelwise=True)


    # NOTES:
    # Layer 0 is useless