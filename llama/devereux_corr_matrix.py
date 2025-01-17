from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from scipy import stats

from llama.devereux_analysis import get_devereux_RSM_by_type, get_devereux_top50_RSM, get_devereux_RSM_by_type_
from llama.devereux_llama import get_dev_explore_BERT, get_deve_llama_RSM
from llama.devereux_w2v import get_w2v_deve_RSM
from llama.model_comparison import plot_heatmap
from llama.model_settings import get_explore_llama


def make_corr_matrix(pf_thresh=300, quick=1):
    llama_3b = get_explore_llama(activation_model='meta-llama/Llama-3.2-3b',
                                attn=False, st=4, end=16)
    llama_3b_RSM = get_deve_llama_RSM(llama_3b, pf_thresh=pf_thresh,
                                   position=0, quick=1)

    llama_70b = get_explore_llama(activation_model=r'meta-llama/Llama-3.3-70b-Instruct',
                                attn=False, st=7, end=30)
    llama_70b_RSM = get_deve_llama_RSM(llama_70b, pf_thresh=pf_thresh,
                                      position=0, quick=1)

    llama_2 = get_explore_llama(activation_model=r'meta-llama/Llama-2-7b-hf',
                                attn=False, st=6, end=20)
    llama_2_RSM = get_deve_llama_RSM(llama_2, pf_thresh=pf_thresh,
                                      position=0, quick=1)

    simCSE_m = get_dev_explore_BERT(bert_type='simCSE', st=2)
    simCSE_RSM = get_deve_llama_RSM(simCSE_m, pf_thresh=pf_thresh, quick=1, )

    BERT_m = get_dev_explore_BERT(bert_type='BERT', st=2)
    BERT_RSM = get_deve_llama_RSM(BERT_m, pf_thresh=pf_thresh, quick=1)

    w2v_RSM = get_w2v_deve_RSM(pf_thresh=pf_thresh)
    model_RSMs = [llama_3b_RSM, llama_70b_RSM, llama_2_RSM,
                  simCSE_RSM, BERT_RSM, w2v_RSM]
    labels = ['Llama 3.2-3b', 'Llama 3.3-70b', 'Llama 2-7b',
              'simCSE', 'BERT', 'word2vec']

    # type2RSM = get_devereux_RSM_by_type(pf_thresh=pf_thresh, attn=False,
    #                                     odd_even=None)
    # type2RSM = get_devereux_top50_RSM(pf_thresh=pf_thresh, )
    # type2RSM_even = get_devereux_top50_RSM(pf_thresh=pf_thresh, odd_even=0)
    # type2RSM_odd = get_devereux_top50_RSM(pf_thresh=pf_thresh, odd_even=1)
    # type2RSM.update(type2RSM_even)
    # type2RSM.update(type2RSM_odd)

    type2RSM = get_devereux_RSM_by_type_(pf_thresh=pf_thresh, attn=False,
                                         odd_even=None, all=True)
    type2RSM_even = get_devereux_RSM_by_type_(pf_thresh=pf_thresh, attn=False,
                                              odd_even=0, all=True)
    type2RSM_odd = get_devereux_RSM_by_type_(pf_thresh=pf_thresh, attn=False,
                                             odd_even=1, all=True)
    type2RSM['Human'] = type2RSM['all']
    type2RSM['Human (even feats)'] = type2RSM_even['all']
    type2RSM['Human (odd feats)'] = type2RSM_odd['all']

    feature_types = ['Human']#, 'Human (even feats)', 'Human (odd feats)']
    human_RSMs = [type2RSM[feature_type] for feature_type in feature_types]

    trils = np.tril_indices_from(llama_3b_RSM, k=-1)
    # for RSM in model_RSMs + human_RSMs:
    #     print(f'{RSM.shape=}')
    # quit()

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
    plot_heatmap(corrs, labels, vmin=0, vmax=0.8, cmap='coolwarm',
                 title='RSM x RSM between different models',
                 labels_horizontal=labels, )


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


def make_feature_x_feature_matrix(pf_thresh=300, ):
    type2RSM = get_devereux_RSM_by_type(pf_thresh=pf_thresh, attn=False,
                                        odd_even=0)
    type2RSM1 = get_devereux_RSM_by_type(pf_thresh=pf_thresh, attn=False,
                                         odd_even=1)
    feature_types = ['visual perceptual', 'encyclopedic',
                     'functional', 'taxonomic',
                     'other perceptual']
    labels0 = [f'{feature_type}_0' for feature_type in feature_types]
    labels1 = [f'{feature_type}_1' for feature_type in feature_types]

    RSMs0 = np.array([type2RSM[feature_type] for feature_type in feature_types])
    RSMs1 = np.array([type2RSM1[feature_type] for feature_type in feature_types])
    RSMs = list(RSMs0) + list(RSMs1)
    RSMs_flats = np.array([RSM[np.tril_indices_from(RSM)] for RSM in RSMs])

    labels = labels0 + labels1
    corrs = np.full((len(RSMs_flats), len(RSMs_flats)), np.nan)
    for i in range(len(RSMs_flats)):
        for j in range(len(RSMs_flats)):
            if i >= j:
                continue
            label0 = labels[i]
            label1 = labels[j]
            # if label0 == label1:

            # if label0[:-1] == label1[:-1]:
            #     continue
            RSMs_flats_i = RSMs_flats[i]
            RSMs_flats_j = RSMs_flats[j]
            nan_idxs = np.isnan(RSMs_flats_i) | np.isnan(RSMs_flats_j)
            RSMs_flats_i = RSMs_flats_i[~nan_idxs]
            RSMs_flats_j = RSMs_flats_j[~nan_idxs]
            corrs[i, j] = stats.spearmanr(RSMs_flats_i, RSMs_flats_j).correlation
            corrs[j, i] = corrs[i, j]
    # corrs[np.isnan(corrs)] = 0
    # corr = stats.spearmanr(RSMs_flats, nan_policy='omit').correlation
    # print(f'Time needed for correlation: {time() - t_st=:.2f} s')
    plot_heatmap(corrs, labels, vmin=0, vmax=0.5, cmap='viridis',
                 title='RSM x RSM between different models')


if __name__ == '__main__':
    # type2RSM = get_devereux_RSM_by_type(pf_thresh=300, attn=False,
    #                                     odd_even=0)
    # quit()
    make_corr_matrix()
    # make_corr_matrix_llama()
    # make_feature_x_feature_matrix()
