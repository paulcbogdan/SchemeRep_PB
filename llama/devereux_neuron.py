from collections import defaultdict

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_llama import get_llama_activations_deve, get_standard_items_list
from llama.get_obj_scn_vecs import process_cat_cat_inner
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from numba import njit
from time import time
from tqdm import tqdm

from llama.numba_regr_test import pairwise_interaction_t_values_proper
from marinate.pkld import pkld


def get_matrix(item0, item1,
               activation_model='meta-llama/Llama-3.2-1b',
               cat='gate_proj_in'):
    cat_, inner = process_cat_cat_inner(cat)

    res, fp = pickle_wrap(get_llama_activations_deve,
                          kwargs={'item0': item0, 'item1': item1,
                                  'activation_model': activation_model},
                          easy_override=False, verbose=-1, dir_branches=100,
                          RAM_cache=False, get_fp=True)
    mat = [res[inner][cat_][layer_name][1] for layer_name in res[inner][cat_]]
    mat = np.nanmean(np.array(mat), axis=1) # average across item split across idxs
    return mat

def get_mat_M(item1, items0, activation_model
              ):
    mats = []
    for item0 in items0:
        mat = get_matrix(item0, item1,
                         activation_model=activation_model)
        mats.append(mat)
    mat = np.nanmean(np.array(mats), axis=0)
    return mat

@pkld
def get_binary_feat_matrix(items, item_std='deve',
                           feature_type='all',
                           pf_thresh=100, threshold=50,
                           req=1, #odd_even=None
                           ):
    _, df = get_standard_items_list(pf_thresh, item_std,
                                    )
    if feature_type != 'all':
        df = df[df['feature type'] == feature_type]
    feat2onehot = {}
    for feature, df_feature in df.groupby('feature'):
        df_cnt = df_feature.groupby('concept')['pf_orig'].sum()
        df_cnt = df_cnt[df_cnt >= req]
        items_w_feature = set(df_cnt.index)
        # items_w_feature = df_feature['concept'].unique()
        l = []
        for item in items:
            l.append(item in items_w_feature)
        num1s = np.sum(l)
        if num1s < threshold:
            continue
        # print(feature)
        # print(df_cnt)
        feat2onehot[feature] = np.array(l)
    # quit()

    # feature2type = df.groupby('feature')['feature type'].first().to_dict()

    # if 'is_noisy_loud' in feat2onehot and 'does_make_sound_a_noise' in feat2onehot:
    #     feat2onehot['is_noisy'] = (feat2onehot['is_noisy_loud'] |
    #                                feat2onehot['does_make_sound_a_noise'])
    #     feature2type['is_noisy'] = 'custom'
    #     del feat2onehot['is_noisy_loud']
    #     del feat2onehot['does_make_sound_a_noise']
    # elif 'is_noisy_loud' in feat2onehot:
    #     feat2onehot['is_noisy'] = feat2onehot['is_noisy_loud']
    #     feature2type['is_noisy'] = 'custom'
    #     del feat2onehot['is_noisy_loud']
    # elif 'does_make_sound_a_noise' in feat2onehot:
    #     feat2onehot['is_noisy'] = feat2onehot['does_make_sound_a_noise']
    #     feature2type['is_noisy'] = 'custom'
    #     del feat2onehot['does_make_sound_a_noise']


    return feat2onehot

def test_interaction(t, mats_all):
    for layer in range(4, t.shape[0]):
        t_vals = t[layer]
        signif = np.abs(t_vals) > 3
        mat_layer = mats_all[:, layer, :]
        mat_layer_signif = mat_layer[:, signif]
        # print(mat_layer_signif.shape)
        print(f'{mat_layer_signif.shape=}')
        nans = np.sum(np.isnan(mat_layer_signif))
        assert nans == 0
        onehot = np.array(onehot, dtype=np.float64)
        onehot = stats.zscore(onehot, nan_policy='omit')
        mat_layer_signif = np.array(mat_layer_signif, dtype=np.float64)
        mat_layer_signif = stats.zscore(mat_layer_signif, axis=0, nan_policy='omit')
        t_vals_interaction = (
            pairwise_interaction_t_values_proper(onehot, mat_layer_signif))
        t_vals_flat = t_vals_interaction[np.tril_indices_from(t_vals_interaction, k=-1)]
        n, _, _ = plt.hist(t_vals_flat.flatten(), bins=100)
        sd = np.nanstd(t_vals_flat.flatten())
        x = np.linspace(-sd * 3, sd * 3, 1000)
        y = stats.norm.pdf(x, loc=0, scale=sd)
        y *= np.max(n) / np.max(y)
        plt.plot(x, y, 'r-', lw=2, label='Normal Distribution')
        plt.show()
    quit()

#pf_thresh=900, quick=50
def do_deve_neuron(pf_thresh=900, quick=5, item_std='mariam',
                   activation_model='meta-llama/Llama-3.2-3b'
                   ):
    items, df = get_standard_items_list(pf_thresh, item_std)
    feat2onehot = pickle_wrap(get_binary_feat_matrix,
                              kwargs={'items': items,
                                      'item_std': item_std,
                                      'pf_thresh': pf_thresh,
                                      'threshold': 100
                                      })

    np.random.seed(0)
    feat2onehot = {
                   'is_brown': feat2onehot['is_brown'],
                   'is_black': feat2onehot['is_black'],
                   'is_green': feat2onehot['is_green'],
                   'is_white': feat2onehot['is_white'],
                   'is_red': feat2onehot['is_red'],
                   'is_made_of_wood': feat2onehot['is_made_of_wood'],
                   }

    mats_all = []
    for i, item1 in tqdm(enumerate(items), desc='Preparing item matrix'):
        items0 = set()
        while len(items0) < quick:
            item0 = items[np.random.randint(len(items))]
            if item0 == item1 or item0 in items0:
                continue
            items0.add(item0)
        items0 = sorted(list(items0))
        mat = pickle_wrap(get_mat_M,
                          kwargs={'item1': item1,
                                  'items0': items0,
                                  'activation_model': activation_model
                                  },
                          verbose=-1, dir_branches=100)
        mats_all.append(mat)
    mats_all = np.array(mats_all)

    # for feat, onehot in feat2onehot.items():
    for feat in ['is_green']:
    # for feat in ['is_red']:
        onehot = feat2onehot[feat]
        mat_feat0 = mats_all[onehot == 0, :, :]
        mat_feat1 = mats_all[onehot == 1, :, :]
        t, _ = stats.ttest_ind(mat_feat0, mat_feat1, axis=0)

        # onehot_black = np.array(feat2onehot['is_black'], dtype=np.float64)
        # mat_feat0_black = mats_all[onehot_black == 0, :, :]
        # mat_feat1_black = mats_all[onehot_black == 1, :, :]
        # t_black, _ = stats.ttest_ind(mat_feat0_black, mat_feat1_black, axis=0)
        # t[np.abs(t_black) < 2.5] = 0

        # t_signif = np.abs(t) > 4
        print(f'{feat=}')
        return t
        print(t_signif.shape)
        quit()
        # print(t_signif.shape)
        # print(t_signif.shape)
        # quit()


        t_flat = t.flatten()

        n, _, _ = plt.hist(t_flat, bins=100, range=(-10, 10))


        x = np.linspace(-10, 10, 1000)
        y = stats.norm.pdf(x, loc=0, scale=1)
        y *= np.max(n) / np.max(y)
        plt.plot(x, y, 'r-', lw=2, label='Normal Distribution')

        # plt.imshow(t, aspect='auto', interpolation='none')
        # plt.colorbar()
        plt.title(f'{feat=}')
        plt.show()
        quit()

    # mat = np.nanmean(np.array(mats_all), axis=0)


    # get_matrix(item1, item0)

if __name__ == '__main__':
    # Y = np.random.randint(0, 2, (100, 1))
    # Y = np.random.normal(size=(100, 1))
    # Y = np.array(Y, dtype=np.float64)
    # # Y = stats.zscore(Y, axis=0, nan_policy='omit')
    # X = np.random.normal(size=(100, 300))
    # # X = stats.zscore(X, axis=0, nan_policy='omit')
    # t_st = time()
    # out = test_interaction_effect(Y, X)
    # print(out.flatten())
    # print(f'Time needed for interaction testing: {time() - t_st:.3f} s')
    #
    # out[np.diag_indices_from(out)] = np.nan
    # n, _, _ = plt.hist(out.flatten(), bins=100)
    # sd = np.nanstd(out.flatten())
    #
    # x = np.linspace(-sd*3, sd*3, 1000)
    # y = stats.norm.pdf(x, loc=0, scale=sd)
    # y *= np.max(n) / np.max(y)
    # plt.plot(x, y, 'r-', lw=2, label='Normal Distribution')
    # plt.show()
    # quit()
    #
    #
    # t_st = time()
    # out = test_interaction_effect(Y, X)
    # print(f'Time needed for interaction testing: {time() - t_st:.3f} s')
    # quit()

    do_deve_neuron()


