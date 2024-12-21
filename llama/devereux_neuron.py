from collections import defaultdict

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_llama import get_llama_activations_deve, get_standard_items_list
from llama.get_obj_scn_vecs import process_cat_cat_inner
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from numba import njit
from time import time

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

def get_mat_M(item1, items0):
    mats = []
    # print(f'{len(items0)=}')
    for item0 in items0:
        mat = get_matrix(item0, item1)
        # mat = pickle_wrap(get_matrix,
        #                   kwargs={'item0': item0,
        #                           'item1': item1}, )
        mats.append(mat)
    mat = np.nanmean(np.array(mats), axis=0)
    return mat

def get_binary_feat_matrix(items, item_std,
                           feature_type='visual perceptual',
                           pf_thresh=100, threshold=50):
    _, df = get_standard_items_list(pf_thresh, item_std)
    df = df[df['feature type'] == feature_type]
    feat2onehot = {}
    for feature, df_feature in df.groupby('feature'):
        items_w_feature = df_feature['concept'].unique()
        l = []
        for item in items:
            l.append(item in items_w_feature)
        num1s = np.sum(l)
        if num1s < threshold:
            continue
        feat2onehot[feature] = np.array(l)
    return feat2onehot

@njit
def extract_diagonal(arr):
    return np.diag(arr)

@njit(fastmath=True, cache=True)
def test_interaction_effect(onehot, X):
    x_col = X.shape[1]
    n_samples = X.shape[0]
    # onehot = onehot[:, 0]
    intercept = np.ones(n_samples)
    out = np.zeros((x_col, x_col))
    # out = np.empty(x_col * (x_col - 1) // 2)
    cnt = 0
    for i in range(x_col):
        for j in range(i):
            X_i = X[:, i]
            X_j = X[:, j]
            X_ij = X_i * X_j
            X_aug = np.vstack((intercept, X_i, X_j, X_ij)).T

            # X_aug = np.hstack((intercept.reshape(-1, 1),
            #                    X_i[:, None], X_j[:, None],
            #                    X_ij[:, None]))

            XTX = np.dot(X_aug.T, X_aug)
            XTy = np.dot(X_aug.T, onehot)
            beta = np.linalg.solve(XTX, XTy)[:, 0]

            # Calculate residuals and residual variance
            y_pred = np.dot(X_aug, beta)
            residuals = onehot - y_pred
            sse = np.sum(residuals ** 2)
            # print(X_aug.shape)
            sigma_squared = sse / (n_samples - X_aug.shape[1])

            # Compute diagonal of (X^T X)^(-1) without full inversion
            # XT_X_inv_diag = np.linalg.inv(XTX).diagonal()
            # XT_X_inv_diag = extract_diagonal(np.linalg.inv(XTX))

            XT_X_inv_diag = extract_diagonal(np.linalg.pinv(XTX))

            # Compute standard errors of beta coefficients
            se_betas = np.sqrt(XT_X_inv_diag * sigma_squared)

            # Compute t-values for coefficients
            t_values = beta / se_betas

            # Calculate residuals and residual variance
            # y_pred = np.dot(X_aug, beta)
            # residuals = onehot - y_pred
            # sse = np.sum(residuals ** 2)
            # sigma_squared = sse / (n_samples - 3 - 1)
            #
            # # Compute standard errors of beta coefficients
            # XT_X_inv = np.linalg.inv(XTX)
            # se_betas = np.sqrt(np.diag(XT_X_inv) * sigma_squared)
            # t_values = beta / se_betas
            # # out[i, j] = beta[3]
            out[i, j] = t_values[1]
            out[j, i] = t_values[1]
    return out


def do_deve_neuron(pf_thresh=900, quick=50, item_std='mariam'):
    items, df = get_standard_items_list(pf_thresh, item_std)
    # feat2onehot = get_binary_feat_matrix(items, item_std,
    #                                      pf_thresh=pf_thresh,
    #                                      threshold=100)
    feat2onehot = pickle_wrap(get_binary_feat_matrix,
                              kwargs={'items': items,
                                      'item_std': item_std,
                                      'pf_thresh': pf_thresh,
                                      'threshold': 100})
    # print(list(feat2onehot))
    # quit()
    # feat_sums = [np.sum(onehot) for onehot in feat2onehot.values()]
    # plt.hist(feat_sums, bins=100)
    # plt.show()
    # quit()
    np.random.seed(0)
    mats_all = []
    for item1 in items:
        items0 = set()
        while len(items0) < quick:
            item0 = items[np.random.randint(len(items))]
            if item0 == item1 or item0 in items0:
                continue
            items0.add(item0)
        mat = pickle_wrap(get_mat_M,
                          kwargs={'item1': item1,
                                  'items0': items0},
                          verbose=-1)
        mats_all.append(mat)
    mats_all = np.array(mats_all)
    print(mats_all.shape)

    for feat, onehot in feat2onehot.items():
        # TODO: study interaction effects

        mat_feat0 = mats_all[onehot == 0, :, :]
        mat_feat1 = mats_all[onehot == 1, :, :]
        t, _ = stats.ttest_ind(mat_feat0, mat_feat1, axis=0)
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


