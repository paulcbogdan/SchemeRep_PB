from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from warnings import warn
from yaml import warnings

import utils
from Utils.pickle_wrap_funcs import pickle_wrap
from connRSA.conn_regress import do_regr_RSA_sn
from llama.color_test import print_colored_list
from llama.model_settings import get_explore_llama, get_explore_BERT, flat_dict_to_nested, get_base_kw
from org_sns import get_sns


def run_layer(kw, fps, big_voxelwise, region,
              easy_override=False, local=True,
              do_tqdm=False, allow_misses=None,
              ctrl=None):
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
            if ctrl is not None:
                if isinstance(ctrl, list):
                    kw['ROIs_ctrl'] = ctrl
                else:
                    kw['ROIs_ctrl'] = [ctrl]
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

def run_one_semantic(semantic, fps_do, ALL_RESULTS, ALL_RESULTS_VALS, target_ROIs,
                     ctrl_contex=False):
    print(f'Run: {semantic=}')

    if isinstance(semantic, bool) or semantic == 'inc':
        llama_cat, llama_layer, obj_scn = None, None, None
    else:
        llama_cat = semantic[1]
        llama_layer = semantic[2]
        obj_scn = semantic[3]
    for i, target_ROI in enumerate(target_ROIs):
        kw = get_base_kw(target_ROI, model=semantic,
                         fps=fps_do, big_voxelwise=False,
                         local=True)
        # THERE IS SOME FUNKINESS SOMETIMES WITH RANDOM LOW
        # THERE MUST BE RUN_LAYER GETTING OVERRIDDEN SOMEHOW
        if ctrl_contex:
            if isinstance(semantic, bool):
                ctrl = None
            elif semantic[1] in ['attn_weights', 'attn_output']:
                ctrl = (semantic[0], 'gate_proj_in', semantic[2],
                        'obj', semantic[4], semantic[5])
            elif isinstance(semantic, list):
                raise ValueError
            else:
                if semantic[3] == 'obj':
                    # if '70b' not in semantic[4]:
                    #     ctrl = None
                    #     warn('70b llama does not allow controlling obj_solo')
                    # else:
                    ctrl = [(semantic[0], semantic[1], semantic[2],
                             'obj_solo', semantic[4], semantic[5]),
                            (semantic[0], semantic[1], semantic[2],
                             'scn', semantic[4], semantic[5]),
                            ]
                else:
                    ctrl = None
        else:
            ctrl = None

        kw['ctrl'] = ctrl

        import zlib
        kw['kw']['pickle_wrap_key'] = zlib.adler32(str(semantic).encode())
        t, vals = pickle_wrap(run_layer, kwargs=kw,
                              verbose=1, easy_override=False,
                              dir_branches=100, )

        ALL_RESULTS[(target_ROI, llama_cat, obj_scn)].append(t)
        ALL_RESULTS_VALS[(target_ROI, llama_cat, obj_scn)].append(vals)


def run_layers_static(obj_scn=True, attn=False, normalize=True,
                      activation_model='70b'):
    if '2-7' in activation_model:
        activation_model = r'meta-llama/Llama-2-7b-hf'
    elif '3b' in activation_model:
        activation_model = 'meta-llama/Llama-3.2-3b'
    else:
        activation_model = 'meta-llama/Llama-3.3-70b-Instruct'
    activation_model = (activation_model, 'bury')

    target_ROIs = ['Occipital', 'ITL', 'Parietal', 'PFC'] #
    ALL_RESULTS = defaultdict(list)
    ALL_RESULTS_VALS = defaultdict(list)
    if isinstance(attn, bool) and attn:
        semantic_l = get_explore_llama(activation_model=activation_model,
                                       attn=True, do_M=False, last_only=False,
                                       include_scn=True, normalize=normalize,
                                       )
        fps_do = 'obj'
    else:
        if obj_scn:
            semantic_l = get_explore_llama(activation_model=activation_model,
                                           attn=attn, do_M=False, last_only=False,
                                           normalize=normalize,
                                           )
            fps_do = 'obj'
        else:
            semantic_l = []
            fps_do = 'non_obj'

        # if '3b' in activation_model:
        # if 'bury' not in activation_model:
        if not ('bury' in activation_model and '70b' in activation_model[0]):
            semantic_l += get_explore_llama(activation_model=activation_model, #'grok_first'),
                                            attn=False, do_M='obj_solo', last_only=False,
                                            normalize=True
                                           )

        semantic_l_BERT = get_explore_BERT('BERT', do_M='obj_solo')
        semantic_l += semantic_l_BERT
        semantic_l_BERT = get_explore_BERT('simCSE', do_M='obj_solo')
        semantic_l += semantic_l_BERT


        w2v_glove_l = [True, 'glove']
        semantic_l += w2v_glove_l

    # fps_do = ['con7_fMRI']

    for semantic in semantic_l:
        try:
            if 'bury' in semantic[4]: # Can't do first layer
                # print('AAAAAAAAAH')
                # print(semantic)
                if semantic[2] == 0:
                    continue
        except TypeError:
            pass
        run_one_semantic(semantic, fps_do, ALL_RESULTS, ALL_RESULTS_VALS,
                         target_ROIs, ctrl_contex=False if isinstance(attn, str) else attn)

    print_all_results(ALL_RESULTS)
    fps_do2title = {'obj': 'Task: Only encoding',
                    'non_obj': 'Task: All but encoding',
                    'all': 'Task: All tasks',
                    ('bl7_fMRI', ): 'Baseline task',
                    ('con7_fMRI', ): 'Conceptual retrieval task',
                    ('vis7_fMRI', ): 'Visual retrieval task'}
    if isinstance(fps_do, list):
        fps_do = tuple(fps_do)
    plot_results_t(ALL_RESULTS, fps_do2title[fps_do])
    # plot_results_M_SE(target_ROIs, ALL_RESULTS_VALS, fps_do2title[fps_do])
    quit()

def plot_results_M_SE(target_ROIs, ALL_RESULTS_VALS, suptitle):
    plt.rcParams.update({'font.size': 20})
    # fig, axs = plt.subplots(len(target_ROIs), 1,
    #                         figsize=(10, 5 * len(target_ROIs)))
    fig, axs = plt.subplots(1, len(target_ROIs),
                            figsize=(5 * len(target_ROIs), 10))
    cnt = 0
    for i, region in enumerate(target_ROIs):
        plt.sca(axs[cnt])
        cnt += 1
        for (key, ar) in ALL_RESULTS_VALS.items():
            # print(f'{key=}')
            # curr_t, p = stats.ttest_1samp(ar, 0, nan_policy='omit', axis=1)
            # print(f'{curr_t=}, {ALL_RESULTS[key]=}')
            if region not in key: continue
            plot_line(ar, key, region, do_labels=i == len(target_ROIs) - 1)

    fig.legend(loc='lower center', ncol=2, frameon=False)
    plt.suptitle(suptitle, fontsize=28)
    plt.subplots_adjust(bottom=0.15, top=0.91, right=0.95, left=0.12,
                        hspace=.4)
    plt.show()

def plot_line(ar_vals, key, region, do_labels=False):
    confidence_interval = np.std(ar_vals, axis=1) / np.sqrt(len(ar_vals))

    # Create the figure and axis
    # plt.figure(figsize=(10, 6))

    # Plot the main line
    t = np.arange(len(ar_vals))
    M = np.mean(ar_vals, axis=1)

    if 'BERT' == key[1]:
        t *= 2
        color = 'chocolate'
        label = 'BERT'
    elif 'simCSE' == key[1]:
        t *= 2
        color = 'orange'
        label = 'simCSE'
    else:
        if key[2] in ['obj_solo', 'obj_M']:
            if key[2] == 'obj_solo':
                label = 'Item embedding\n(object solo)'
            else:
                label = f'Item embedding\n(scene averages)'
            color = 'red'
        else:
            label = 'Item embedding\n(scene → object)'
            color = 'dodgerblue'
    label = label if do_labels else None

    plt.plot(t, M, color, label=label,
             linewidth=3, marker='o')

    # Plot the confidence interval
    plt.fill_between(t,
                     M - confidence_interval,
                     M + confidence_interval,
                     color=color, alpha=0.2,
                     # label='Confidence Interval'
                     )

    # Customize the plot
    plt.grid(True, alpha=0.3)
    plt.title(region)
    plt.yticks([0, 0.01, 0.02])
    plt.ylim(0, 0.02)
    plt.ylabel('Correlation (r)')
    if do_labels:
        plt.xlabel('Layer')
    plt.gca().spines[['top', 'right', ]].set_visible(False)


def plot_results_t(all_results, subtitle=''):
    nested_dict = flat_dict_to_nested(all_results)
    # print(all_results)
    # quit()
    plt.rcParams.update({'font.size': 20})
    # fig, axs = plt.subplots(len(nested_dict), 1,
    #                         figsize=(10, 5 * len(nested_dict)))
    fig, axs = plt.subplots(1, len(nested_dict),
                            figsize=(5 * len(nested_dict), 5))
    module_type2label = {'gate_proj_in': 'Item embedding',
                         'attn_weights': 'Attention weights',}
    # print(f'{nested_dict=}')
    for j, (region, b_dict) in enumerate(nested_dict.items()):
        try:
            plt.sca(axs[j])
        except TypeError:
            pass
        plt.title(region)
        high = 0
        for i, (module_type, c_dict) in enumerate(b_dict.items()):
            if module_type is None:
                val = c_dict[None][0]
                label = 'word2vec' if j == 0 else None
                plt.plot([14], [val], label=label, color='red',
                          marker='o', linewidth=3, markersize=8)
                plt.plot([13.1, 14.9], [val, val], color='red',
                         linewidth=3)
                continue
            elif module_type == 'l':
                val = c_dict['v'][0]
                label = 'GloVe' if j == 0 else None
                plt.plot([14], [val], label=label, color='crimson',
                          marker='o', linewidth=3, markersize=8)
                plt.plot([13.1, 14.9], [val, val], color='crimson',
                         linewidth=3)
                continue

            if 'simCSE' in module_type or 'BERT' in module_type:
                label = module_type
                if 'obj' in c_dict:
                    values = c_dict['obj']
                    color = 'orange'
                    layer_nums = np.array(list(range(len(values))))# * 2
                    if 'obj_M' in c_dict:
                        label = label if j == 0 else None
                    else:
                        label = f'{label}\n(scene → object)' if j == 0 else None
                    color = 'chocolate' if module_type == 'BERT' else 'orange'
                    plt.plot(layer_nums, values, label=label,
                             color=color, marker='o',
                             # color='chocolate' if module_type == 'BERT' else 'orange',
                             linewidth=3)
                    high = np.max([high, np.max(values)])

                if 'obj_M' in c_dict:
                    color = 'chocolate'
                    values = c_dict['obj_M']
                    layer_nums = np.array(list(range(len(values))))# * 2
                    if 'obj' in c_dict:
                        label = f'{label}\n(scene averages)' if j == 0 else None
                    else:
                        label = f'{label}' if j == 0 else None
                    color = 'chocolate' if module_type == 'BERT' else 'orange'
                    plt.plot(layer_nums, values, label=label,
                             color=color, marker='o',
                             # color='chocolate' if module_type == 'BERT' else 'orange',
                             linewidth=3)
                    high = np.max([high, np.max(values)])
            elif 'obj' in c_dict or 'obj_dif' in c_dict:
                # print(f'{c_dict=}')
                if module_type == 'attn_weights':
                    label = 'Llama-3.2-3b\n(attention weights)'
                elif module_type == 'gate_proj_in':
                    label = 'Llama-3.2-3b\n(contextualized embedding)'
                elif module_type == 'attn_output':
                    label = 'Llama-3.2-3b\n(attention output)'
                elif module_type == 'down_proj_out':
                    label = 'Llama-3.2-3b\n(MLP output)'
                elif module_type == 'input':
                    label = 'Llama-3.2-3b\n(residual input)'
                else:
                    raise ValueError
                if 'obj' in c_dict:
                    values = c_dict['obj']
                else:
                    values = c_dict['obj_dif']
                layer_nums = np.array(list(range(len(values))))
                color = 'green' if 'attn' in module_type else 'purple'
                assert len(values) != 56
                plt.plot(layer_nums, values, label=label if j == 0 else None,
                         color=color, linewidth=3, marker='o',)
                high = np.max([high, np.max(values)])
            elif 'scn' in c_dict:
                print('TOAST')
                if module_type == 'attn_weights':
                    label = 'Attention weights (scene)'
                elif module_type == 'gate_proj_in':
                    label = 'Scene embedding\n(object → scene)'
                else:
                    raise ValueError
                # if 'obj' in c_dict:
                #     values = c_dict['obj']
                # else:
                values = c_dict['scn']
                layer_nums = np.array(list(range(len(values))))
                # color = 'purple' if 'attn' in module_type else 'dodgerblue'
                plt.plot(layer_nums, values, label=label if j == 0 else None,
                         color='orange', linewidth=3, marker='o')
                high = np.max([high, np.max(values)])

            if ('obj_M' in c_dict and not
                ('simCSE' in module_type or 'BERT' in module_type)):
                label = 'Item embedding\n(object; scene averages)'
                values = c_dict['obj_M']
                layer_nums = np.array(list(range(len(values))))
                plt.plot(layer_nums, values, label=label if j == 0 else None,
                         color='red', linewidth=3, marker='o')
                high = np.max([high, np.max(values)])
            if 'obj_solo' in c_dict:
                label = 'Llama-3.2-3b\n(static embedding)'
                values = c_dict['obj_solo']
                layer_nums = np.array(list(range(len(values))))
                plt.plot(layer_nums, values, label=label if j == 0 else None,
                         color='dodgerblue', linewidth=3, marker='o')
                high = np.max([high, np.max(values)])

        plt.yticks([0, 2, 4, 6, 8, 10, 12])

        if high > 8:
            plt.ylim(0, 13.2)#high * 1.1)
        else:
            plt.ylim(0, 8.8)
        if j == 0:
            plt.ylabel('t-value')
        # print(f'{b_dict=}')
        if 'attn_weights' in b_dict:
            plt.xlabel('Layer')
        else:
            plt.xlabel('Layer', labelpad=-53)
        plt.gca().spines[['top', 'right', ]].set_visible(False)

    # plt.legend(['a', 'b', 'c', 'd', 'e'], [0, 1, 2, 3, 4])
        if j == 0:
            handles, labels = plt.gca().get_legend_handles_labels()
            # handles = handles[1:] + handles[:1]
            # labels = labels[1:] + labels[:1]
            # plt.legend(handles, labels)
    #
    # # Sort them using a paired sort (sorts labels and reorders handles accordingly)
    # sorted_pairs = sorted(zip(labels, handles))
    # sorted_labels, sorted_handles = zip(*sorted_pairs)
    #
    # # Create legend with sorted labels
    # plt.legend(sorted_handles, sorted_labels)

    # plt.xlim(-0.5, 28.5)

    legend = fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False,
               # title_fontproperties={'ha': 'center'}
                       )
    for text in legend.get_texts():
        text.set_ha('center')
    # if len(b_dict) == 1:
    #     plt.subplots_adjust(bottom=0.21, top=0.91, right=0.95, left=0.04,
    #                         hspace=.4)
    # else:
    plt.subplots_adjust(bottom=0.28, top=0.91, right=0.98, left=0.04,
                        hspace=.4)
    # plt.tight_layout()
    # plt.suptitle(subtitle, fontsize=28)
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
    run_layers_static()
