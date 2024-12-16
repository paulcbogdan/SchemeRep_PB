import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

from llama.get_obj_scn_vecs import get_sn_fp_llama_RSM_l, get_sn_fp_llama_RSM
from scipy import stats

def run_RSM_x_RSM():
    activation_model = 'meta-llama/Llama-3.2-3b'
    activation_model = 'meta-llama/Llama-3.1-70b'
    activation_model = 'meta-llama/Llama-3.3-70b-Instruct'

    normalize = True
    all_llama_layers = list(range(0, 80))

    all_llama_cats = ['gate_proj_in', 'down_proj_in',
                      'gate_proj_out', 'up_proj_out',
                      # 'down_proj_out', # bit of an outlier in heterogeneity
                      'act_fn_out',
                      #'q_proj', #'k_proj',
                      #'v_proj',
                      'attn_weights',
                      # 'attn_output',
                      'input'
                      ]

    all_llama_cats = ['gate_proj_in']#, 'attn_weights']
    # all_llama_cats = ['gate_proj_out']
    # all_llama_cats = ['attn_weights']
    semantic_l = []
    cnt = 0
    tick_lows = []
    tick_mids = []
    RSM_stims_all = []
    ticks = []
    for llama_layer in all_llama_layers:
        tick_lows.append(cnt)
        ticks.append(llama_layer)
        for llama_cat in all_llama_cats:

    # for llama_cat in all_llama_cats:
    #     ticks.append(llama_cat)
    #     tick_lows.append(cnt)
    #     for llama_layer in all_llama_layers:
            semantic = ('llama', llama_cat, llama_layer, 'scn', activation_model,
                        normalize)
            RSM_stim = get_sn_fp_llama_RSM(104, 'obj7_fMRI', semantic)
            RSM_stim = RSM_stim[np.tril_indices_from(RSM_stim, k=-1)]
            RSM_stims_all.append(RSM_stim)
            cnt += 1
        tick_mid = (cnt - tick_lows[-1]) // 2 + tick_lows[-1]
        tick_mids.append(tick_mid)

        #


    RSM_stims_all = np.array(RSM_stims_all)
    nan_cols = np.isnan(RSM_stims_all).all(axis=0)
    RSM_stims_all = RSM_stims_all[:, ~nan_cols]
    # print(f'{np.mean(nan_cols)=}')
    print(RSM_stims_all.shape)

    plt.figure(figsize=(12, 12))
    # set default font size to 16
    plt.rcParams.update({'font.size': 18})
    # print(RSM_stims_all.shape)
    # quit()
    RSM_stims_all = stats.rankdata(RSM_stims_all, axis=1)

    corr = spatial.distance.pdist(RSM_stims_all, 'correlation')
    corr = spatial.distance.squareform(corr)
    corr[np.diag_indices_from(corr)] = np.nan
    corr = 1 - corr
    plt.imshow(corr, cmap='turbo', vmin=0.6, vmax=1.0, interpolation='none')

    if len(tick_mids) > 20:
        tick_mids = tick_mids[::5]
        ticks = ticks[::5]

    # plt.title()
    plt.xticks(tick_mids, ticks, rotation=90, fontsize=16)
    plt.xlabel('Layer')
    plt.yticks(tick_mids, ticks, rotation=0, fontsize=16)
    plt.ylabel('Layer')
    ax = plt.colorbar()
    ax.set_label('Spearman (r)')
    plt.title('RSM x RSM between layers\' MLP outputs')
    # plt.tight_layout()
    plt.show()

    return semantic_l
    



if __name__ == '__main__':
    run_RSM_x_RSM()
