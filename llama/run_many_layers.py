from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

import utils
from Utils.pickle_wrap_funcs import pickle_wrap
from connRSA.conn_regress import do_regr_RSA_sn
from llama.color_test import print_colored_list
from llama.model_settings import get_explore_llama, get_explore_BERT, flat_dict_to_nested, get_base_kw
from org_sns import get_sns


def run_layer(kw, fps, big_voxelwise, region,
              easy_override=False, local=True,
              do_tqdm=False, allow_misses=None):
    if local:
        assert not big_voxelwise, 'Cannot be local and big_voxelwise'
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    bad_sns_euc = ['132', '138', '224', '234']
    if kw['trial_similarity'] == 'euc':
        sns = [sn for sn in sns if sn not in bad_sns_euc]
    sns = [sn for sn in sns if sn not in bad_sns]

    vals = np.full((len(sns), len(fps)), np.nan)
    if do_tqdm:
        loop = tqdm(sns, desc='looping through subjects')
    else:
        loop = sns
    for i, sn in enumerate(loop):
        for j, fp in enumerate(fps):
            kw['sn'] = sn
            kw['fp'] = fp
            if local:
                kw['ROI_focus'] = f'{region}_M_corr'
            elif big_voxelwise:
                kw['ROI_focus'] = f'{region}_BOLD_cmb'
            else:
                kw['ROI_focus'] = f'{region}_BOLD'
            kw['ROIs_ctrl'] = []
            kw['return_dif'] = True
            if allow_misses is not None and (sn in allow_misses or allow_misses == 'all'):
                try:
                    corr, _, _ = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                                   verbose=-1,
                                                   easy_override=easy_override,
                                                   dir_branches=100)
                except FileNotFoundError:
                    print(f'Missing file for: {sn}, {fp=}')
                    vals[i, j] = np.nan
                    continue
            else:
                corr, _, _ = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                               verbose=-1,
                                               easy_override=easy_override,
                                               dir_branches=100)
            vals[i, j] = corr

    if allow_misses is not None:
        vals = np.nanmean(vals, axis=1)
        # assert np.sum(np.isnan(vals)) == 0
    else:
        assert np.sum(np.isnan(vals)) == 0
        vals = np.mean(vals, axis=1)

    if len(fps) > 1:
        title = 'All tasks: '
    else:
        mapper = {'bl7_fMRI': 'Baseline task', 'obj7_fMRI': 'Encoding task',
                  'con7_fMRI': 'Conceptual retrieval', 'vis7_fMRI': 'Visual retrieval'}
        title = f'{mapper[fps[0]]}: '

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
    print(f'\n- {title.replace("\n", " ")} -')
    t, p = stats.ttest_1samp(vals, 0, nan_policy='omit' if allow_misses else 'raise')
    N = len(sns)
    local_str = 'Local' if local else 'Dist. voxelwise' if big_voxelwise else 'Dist averages'
    print(f'{local_str}: t[{N - 1}] = {t:.2f}, p = {p:.3f}')
    return t, vals


def run_many_layers(big_voxelwise=True):
    target_ROIs = ['Occipital', 'ITL', 'Parietal', 'PFC']

    ALL_RESULTS = defaultdict(list)

    target_roi2baddies = defaultdict(list)
    semantic_l = get_explore_BERT()
    semantic_l = [True]
    semantic_l = get_explore_llama(activation_model=r'meta-llama/Llama-2-7b-hf',
                                   attn=False)
    # semantic_l = get_explore_llama(activation_model=r'meta-llama/Llama-3.3-70b-Instruct',
    #                                attn=False)
    semantic_l = get_explore_llama(activation_model=(r'meta-llama/Llama-3.2-3b', 'grok_first'),
                                   attn=False)

    # semantic_l = get_explore_llama(activation_model=r'meta-llama/Llama-3.3-70b-Instruct',
    #                                attn=False)
    semantic_l = semantic_l[4::5]

    for semantic in semantic_l:
        if isinstance(semantic, bool) and semantic:
            llama_cat, llama_layer, obj_scn = None, None, None
        else:
            llama_cat = semantic[1]
            llama_layer = semantic[2]
            obj_scn = semantic[3]
        for i, target_ROI in enumerate(target_ROIs):
            kw = get_base_kw(target_ROI, model=semantic,
                             fps='obj', big_voxelwise=big_voxelwise)
            # THERE IS SOME FUNKINESS SOMETIMES WITH RANDOM LOW
            # THERE MUST BE RUN_LAYER GETTING OVERRIDDEN SOMEHOW
            t, _ = pickle_wrap(run_layer, kwargs=kw,
                            verbose=-1, easy_override=True,
                            dir_branches=100)

            ALL_RESULTS[(target_ROI, llama_cat, obj_scn)].append(t)
        print_all_results(ALL_RESULTS)
    target_roi2baddies = dict(target_roi2baddies)
    print(f'{target_roi2baddies=}')
    plot_all_results(ALL_RESULTS)
    quit()


def plot_all_results(all_results):
    nested_dict = flat_dict_to_nested(all_results)
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(len(nested_dict), 1, figsize=(10, 5 * len(nested_dict)))
    for j, (region, b_dict) in enumerate(nested_dict.items()):
        print(f'Region: {region}')
        plt.sca(axs[j])
        plt.title(region)
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
    fig.legend(loc='lower center', ncol=3)
    plt.tight_layout()
    plt.show()



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
    run_many_layers(big_voxelwise=True)
