from Utils.pickle_wrap_funcs import pickle_wrap
import inspect

from llama.model_settings import get_base_kw, get_explore_llama, get_explore_BERT
from llama.run_many_layers import run_layer
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def retrieve_name(v):
    if isinstance(v, bool) and v:
        return ['word2vec']
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is v]

def compare_depths(region='PFC',
                   # activation_model='meta-llama/Llama-3.3-70b-Instruct',
                   activation_model='meta-llama/Llama-3.2-3b',
                   big_voxelwise=False, local=True,
                   attn=False):
    if activation_model == 'meta-llama/Llama-3.3-70b-Instruct':
        low = 8
        high = 80
        tick = 8
    else:
        low = 4
        high = 28
        tick = 4

    models = [get_explore_llama(activation_model=activation_model,
                                normalize=True, attn=attn, st=x, end=x + tick,
                                do_M=True)
              for x in range(low, high, tick)]
    model_strs = [f'depth {x}-{x+tick}' for x in range(low, high, tick)]
    model_vals = run_models(models, model_strs, region, big_voxelwise, local)
    t_mat = cross_t(model_vals)
    plot_heatmap(t_mat, model_strs, region, attn)


def compare_llamas(region='PFC',
                   # activation_model='meta-llama/Llama-3.3-70b-Instruct',
                   activation_model='meta-llama/Llama-3.2-3b',
                   big_voxelwise=False, local=True):

    # 3.2 3b best at: v_proj: t=7.924

    all_llama_cats = ['gate_proj_in', 'down_proj_in', 'gate_proj_out',
                      'up_proj_out', 'down_proj_out', 'act_fn_out',
                      'q_proj', 'k_proj', 'v_proj', 'attn_weights',
                      'attn_output', 'input'
                      ]
    models = [get_explore_llama(activation_model=activation_model,
                                normalize=True, attn=s, st=4)
              for s in all_llama_cats]
    model_strs = all_llama_cats

    model_vals = run_models(models, model_strs, region, big_voxelwise, local)
    t_mat = cross_t(model_vals)
    plot_heatmap(t_mat, model_strs, region, attn)

def compare_models(region='PFC', big_voxelwise=True, local=False,
                   # attn='all_minus_attn'
                   attn=False
                   ):
    llama2_7b = get_explore_llama(activation_model=r'meta-llama/Llama-2-7b-hf',
                                  attn=attn, st=8)
    llama32_3b = get_explore_llama(activation_model='meta-llama/Llama-3.2-3b',
                                   attn=attn, st=8, do_M=False)
    # llama33_70 = get_explore_llama(activation_model=r'meta-llama/Llama-3.3-70b-Instruct',
    #                                attn=attn, st=8)
    # llama33_70 = get_explore_llama(activation_model=(r'meta-llama/Llama-3.2-3b', 'grok_first'),
    #                                attn=attn, st=8)
    llama32_3b_M = get_explore_llama(activation_model='meta-llama/Llama-3.2-3b',
                                     attn=attn, st=8, do_M=True)
    # llama33_70 = get_explore_llama(activation_model=r'meta-llama/Llama-3.1-70b',
    #                                attn=attn, st=8)
    BERT = get_explore_BERT('BERT', st=2)
    simCSE = get_explore_BERT('simCSE', st=2)
    word2vec = True

    # models = [llama2_7b, llama31_3b, llama33_70, BERT, simCSE, word2vec]
    models = [llama32_3b, llama32_3b_M, BERT, simCSE, word2vec]
    model_strs = [retrieve_name(model)[0] for model in models]
    model_vals = run_models(models, model_strs, region, big_voxelwise, local)
    t_mat = cross_t(model_vals)
    plot_heatmap(t_mat, model_strs, region, attn)

def run_models(models, model_strs, region, big_voxelwise, local):
    model_vals = []
    for model, model_str in zip(models, model_strs):
        kw = get_base_kw(region, model=model,
                         fps='non_obj', big_voxelwise=big_voxelwise,
                         local=local)
        t, vals = pickle_wrap(run_layer, kwargs=kw,
                              verbose=-1, easy_override=False,
                              dir_branches=100)
        d = t / np.sqrt(60)
        print(f'{model_str}: {t=:.3f}, {d=:.3f}')
        model_vals.append(vals)
    return model_vals

def cross_t(model_vals):
    t_mat = np.zeros((len(model_vals), len(model_vals)))
    for i, vals0 in enumerate(model_vals):
        for j, vals1 in enumerate(model_vals):
            if i == j:
                continue
            t, p = stats.ttest_rel(vals0, vals1)
            t_mat[i, j] = t
    return t_mat

def plot_heatmap(corr_t, region_labels, region=None, attn=None,
                 vmin=-5, vmax=5, cmap='coolwarm', title=None):
    fig, ax = plt.subplots()
    im = ax.imshow(corr_t, cmap=cmap, interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)

    # Add text annotations
    for i in range(len(region_labels)):
        for j in range(len(region_labels)):
            if np.isnan(corr_t[i, j]):
                continue
            if cmap == 'coolwarm':
                color = 'k'
            else:
                color = 'k' if corr_t[i, j] > .1 else 'w'
            text = ax.text(j, i, f"{corr_t[i, j]:.2f}",
                           ha="center", va="center", color=color)

    # Set the ticks and labels
    if region is not None:
        attn_str = 'attn' if attn else 'item'
        plt.title(f'{region}: {attn_str}')
    ax.set_xticks(np.arange(len(region_labels)),
                 )
    ax.set_yticks(np.arange(len(region_labels)))
    ax.set_xticklabels(region_labels,
                       rotation=50)
    ax.set_yticklabels(region_labels)

    # Set the title and show the plot
    # screen_title = screen2name()[screen]
    # ax.set_title(f"{screen_title}: Correlation Matrix\n"
    #              f"(region = {region})")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # compare_models()
    # compare_llamas()
    compare_depths()
