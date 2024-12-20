import numpy as np

from llama.devereux_analysis import get_devereux_RSM_by_type
from llama.devereux_llama import get_dev_explore_BERT, get_deve_llama_RSM
from llama.devereux_w2v import get_w2v_deve_RSM
from llama.model_comparison import retrieve_name, plot_heatmap
from llama.model_settings import get_explore_llama

import matplotlib.pyplot as plt
from scipy import stats
from time import time
from scipy import spatial


def make_corr_matrix(pf_thresh=300, quick=None):
    llama_m = get_explore_llama(activation_model='meta-llama/Llama-3.2-3b',
                                attn=False, st=2)
    llama_RSM = get_deve_llama_RSM(llama_m, pf_thresh=pf_thresh, quick=quick)

    simCSE_m = get_dev_explore_BERT(bert_type='simCSE', st=2)
    simCSE_RSM = get_deve_llama_RSM(simCSE_m, pf_thresh=pf_thresh, quick=quick)

    BERT_m = get_dev_explore_BERT(bert_type='BERT', st=2)
    BERT_RSM = get_deve_llama_RSM(BERT_m, pf_thresh=pf_thresh, quick=quick)

    w2v_RSM = get_w2v_deve_RSM(pf_thresh=pf_thresh)
    model_RSMs = [llama_RSM, simCSE_RSM, BERT_RSM, w2v_RSM]
    labels = ['Llama', 'simCSE', 'BERT', 'word2vec']

    type2RSM = get_devereux_RSM_by_type(pf_thresh=pf_thresh, attn=False)

    feature_types = ['visual perceptual', 'encyclopedic',
                     'functional', 'taxonomic',
                     'other perceptual']
    human_RSMs = [type2RSM[feature_type] for feature_type in feature_types]

    trils = np.tril_indices_from(llama_RSM, k=-1)

    RSMs_flats = np.array([RSM[trils] for RSM in model_RSMs + human_RSMs])
    t_st = time()
    corrs = np.full((len(RSMs_flats), len(RSMs_flats)), np.nan)
    for i in range(len(RSMs_flats)):
        for j in range(len(RSMs_flats)):
            if i >= j:
                continue
            RSMs_flats_i = RSMs_flats[i]
            RSMs_flats_j = RSMs_flats[j]
            nan_idxs = np.isnan(RSMs_flats_i) | np.isnan(RSMs_flats_j)
            RSMs_flats_i = RSMs_flats_i[~nan_idxs]
            RSMs_flats_j = RSMs_flats_j[~nan_idxs]
            corrs[i, j] = stats.spearmanr(RSMs_flats_i, RSMs_flats_j).correlation
            corrs[j, i] = corrs[i, j]


    labels = labels + feature_types
    # corr = stats.spearmanr(RSMs_flats, nan_policy='omit').correlation
    print(f'Time needed for correlation: {time() - t_st=:.2f} s')
    plot_heatmap(corrs, labels, vmin=0, vmax=0.25, cmap='viridis',
                 title='RSM x RSM between different models')

def make_corr_matrix_llama(pf_thresh=300, quick=None):
    llama_ms = get_explore_llama(activation_model='meta-llama/Llama-3.2-3b',
                                attn=False, st=2)
    llama_RSMs = [get_deve_llama_RSM(llama_m, pf_thresh=pf_thresh, quick=quick)
                  for llama_m in llama_ms]
    trils = np.tril_indices_from(llama_RSMs[0], k=-1)
    llama_RSMs = np.array([llama_RSM[trils] for llama_RSM in llama_RSMs])


    llama_RSMs = stats.rankdata(llama_RSMs, axis=1)
    corr = spatial.distance.pdist(llama_RSMs, 'correlation')
    corr = spatial.distance.squareform(corr)
    corr = 1 - corr

    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 18})
    plt.imshow(corr, cmap='turbo', vmin=0.4, vmax=1.0,
               interpolation='none')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    ax = plt.colorbar()
    ax.set_label('Spearman (r)', rotation=270, labelpad=25)
    plt.title('RSM x RSM between Llama layers', pad=15)
    plt.show()


if __name__ == '__main__':
    # make_corr_matrix()
    make_corr_matrix_llama()
