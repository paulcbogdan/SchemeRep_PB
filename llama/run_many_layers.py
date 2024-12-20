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
              do_tqdm=False, allow_misses=None,
              ):
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
                                                   dir_branches=100,
                                                   )
                except FileNotFoundError:
                    print(f'Missing file for: {sn}, {fp=}')
                    vals[i, j] = np.nan
                    continue
            else:
                (corr, _, _), fp_pkl = utils.pickle_wrap(do_regr_RSA_sn, kwargs=kw,
                                               verbose=-1,
                                               easy_override=easy_override,
                                               dir_branches=100, get_fp=True)
                # print(f'{fp_pkl=}')
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


def run_many_layers(big_voxelwise=True, attn=False):
    target_ROIs = ['Occipital', 'ITL', 'PFC'] # 'Parietal',

    ALL_RESULTS = defaultdict(list)

    target_roi2baddies = defaultdict(list)
    semantic_l = get_explore_BERT()
    semantic_l = [True]
    semantic_l = get_explore_llama(activation_model=r'meta-llama/Llama-2-7b-hf',
                                   attn=False)
    # semantic_l = get_explore_llama(activation_model=r'meta-llama/Llama-3.3-70b-Instruct',
    #                                attn=False)
    # semantic_l = get_explore_llama(activation_model=(r'meta-llama/Llama-3.2-3b', 'grok_first'),
    #                                attn=False)
    # semantic_l = get_explore_llama(activation_model=(r'meta-llama/Llama-3.2-3b', 'grok_first'),
    #                                attn=False)
    semantic_l = get_explore_llama(activation_model=r'meta-llama/Llama-3.2-3b', #'grok_first'),
                                   attn=attn, do_M=False)
    if not attn:
        semantic_l_ = get_explore_llama(activation_model=r'meta-llama/Llama-3.2-3b', #'grok_first'),
                                       attn=attn, do_M=True)
    # semantic_l += semantic_l_
    # semantic_l += get_explore_BERT('BERT') + get_explore_BERT('simCSE')
    # semantic_l = get_explore_BERT('simCSE')

    # semantic_l = semantic_l[3::4]
    # semantic_l = semantic_l[::-1]
    fps_do = 'non_obj'
    # fps_do = 'obj'

    for semantic in semantic_l:
        if isinstance(semantic, bool) and semantic:
            llama_cat, llama_layer, obj_scn = None, None, None
        else:
            llama_cat = semantic[1]
            llama_layer = semantic[2]
            obj_scn = semantic[3]
        for i, target_ROI in enumerate(target_ROIs):
            kw = get_base_kw(target_ROI, model=semantic,
                             fps=fps_do, big_voxelwise=big_voxelwise)
            # print(kw)
            # quit()
            # THERE IS SOME FUNKINESS SOMETIMES WITH RANDOM LOW
            # THERE MUST BE RUN_LAYER GETTING OVERRIDDEN SOMEHOW
            # kw['easy_override'] = True
            # kw['model'] = semantic
            import zlib
            kw['kw']['pickle_wrap_key'] = zlib.adler32(str(semantic).encode())
            t, _ = pickle_wrap(run_layer, kwargs=kw,
                            verbose=-1, easy_override=True,
                            dir_branches=100, )
            # print((target_ROI, llama_cat, obj_scn))

            ALL_RESULTS[(target_ROI, llama_cat, obj_scn)].append(t)
    print_all_results(ALL_RESULTS)
    fps_do2title = {'obj': 'Task: Only encoding', 'non_obj': 'Task: All but encoding',
                    'all': 'Task: All tasks'}
    plot_all_results(ALL_RESULTS, fps_do2title[fps_do])


def plot_all_results(all_results, subtitle=''):
    nested_dict = flat_dict_to_nested(all_results)
    # print(all_results)
    # quit()
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(len(nested_dict), 1, figsize=(10, 5 * len(nested_dict)))
    module_type2label = {'gate_proj_in': 'Item embedding',
                         'attn_weights': 'Attention weights',}
    for j, (region, b_dict) in enumerate(nested_dict.items()):
        plt.sca(axs[j])
        plt.title(region)
        high = 0
        for i, (module_type, c_dict) in enumerate(b_dict.items()):
            if 'simCSE' in module_type or 'BERT' in module_type:
                label = module_type
                values = c_dict['obj']
                layer_nums = np.array(list(range(len(values)))) * 2
                plt.plot(layer_nums, values, label=label if j == 0 else None,
                         color='olive' if module_type == 'BERT' else 'limegreen',
                         linewidth=3)
                high = np.max([high, np.max(values)])

            elif 'obj' in c_dict:
                print('TOAST')
                if module_type == 'attn_weights':
                    label = 'Attention weights'
                elif module_type == 'gate_proj_in':
                    label = 'Item embedding\n(scene → object)'
                else:
                    raise ValueError
                values = c_dict['obj']
                layer_nums = np.array(list(range(len(values))))
                color = 'purple' if 'attn' in module_type else 'dodgerblue'
                plt.plot(layer_nums, values, label=label if j == 0 else None,
                         color=color, linewidth=3)
                high = np.max([high, np.max(values)])

            if 'obj_M' in c_dict:
                label = 'Item embedding\n(object; scene averages)'
                values = c_dict['obj_M']
                layer_nums = np.array(list(range(len(values))))
                plt.plot(layer_nums, values, label=label if j == 0 else None,
                         color='red', linewidth=3)
                high = np.max([high, np.max(values)])

        plt.yticks([0, 2, 4, 6, 8, 10, 12])
        plt.ylabel('t-value')
        plt.ylim(0, high * 1.1)
        plt.ylim(0, 7)
        # if j == 0:
        #     plt.legend(frameon=False, ncol=2)
        if j == len(nested_dict) - 1:
            plt.xlabel('Layer')
        plt.gca().spines[['top', 'right', ]].set_visible(False)

    fig.legend(loc='lower center', ncol=2, frameon=False)
    if len(b_dict) == 1:
        plt.subplots_adjust(bottom=0.11, top=0.91, right=0.95, left=0.08,
                            hspace=.4)
    else:
        plt.subplots_adjust(bottom=0.18, top=0.91, right=0.95, left=0.08,
                            hspace=.4)
    # plt.tight_layout()
    plt.suptitle(subtitle, fontsize=28)
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
