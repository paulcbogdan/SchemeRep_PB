import pdb

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_get_quick_lists import generate_comparison_pairs_d
from llama.devereux_llama import get_standard_items_list
from llama.devereux_neuron import get_mat_M


# Import necessary libraries
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeClassifier, RidgeClassifierCV
from sklearn.datasets import make_regression
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_item_vecs(items, quick=1,
                  activation_model='meta-llama/Llama-3.2-3b'):
    items_sorted = sorted(items)
    item2pair = generate_comparison_pairs_d(tuple(items_sorted), quick)
    item2mat = {}
    for item in tqdm(items_sorted, desc='Getting item vecs'):
        pairs = item2pair[item]
        print(f'{item}: {pairs=}')
        mat = pickle_wrap(get_mat_M,
                          kwargs={'item1': item,
                                  'items0': pairs,
                                  'activation_model': activation_model
                                  },
                          verbose=-1, dir_branches=100)
        item2mat[item] = mat
    mats_all = [item2mat[item] for item in items]
    mats_all = np.array(mats_all)
    return mats_all

def test_regression_models(X, y, items):
    num_nans = np.isnan(X).sum()
    num_infs = np.isinf(X).sum()
    total = X.size
    # print(f"NaNs: {num_nans}/{total} ({num_nans / total:.2f})")
    # print(f"Infs: {num_infs}/{total} ({num_infs / total:.2f})")
    assert num_nans == 0
    assert num_infs == 0

    cols_all_same = (X.std(axis=0) == 0).sum()
    print(f"Cols all same: {cols_all_same}/{X.shape[1]}")
    X = X[:, X.std(axis=0) != 0]
    X = stats.zscore(X, axis=0)

    ridge = RidgeClassifier(alpha=1.0, class_weight='balanced')  # Regularization parameter

    # Using StratifiedKFold to maintain the proportion of labels in folds
    skf = StratifiedKFold(n_splits=5)
    y_decision = cross_val_predict(ridge, X, y, cv=skf, method='decision_function')
    for i in range(y_decision.shape[0]):
        y_actual = y[i]
        if y_decision[i] > 0:
            y_actual_str = '(correct)' if y_actual == 1 else ''
            print(f'{items[i]}: {y_decision[i]:.2f} {y_actual_str}')
        elif y_actual > 0:
            print(f'{items[i]}: {y_decision[i]:.2f} (miss)')
    quit()

    # Convert continuous predictions to binary labels
    y_pred = (y_decision > 0).astype(int)

    # Generate classification report
    report = classification_report(y, y_pred, target_names=["Class 0", "Class 1"])

    # print("Continuous Decision Function Values:\n")
    # print(y_decision)
    print("\nClassification Report:\n")
    print(report)
    quit()


def get_binary_feature_matrix_single(pf_thresh=900,
                                     # feature='does_protect',
                                     feature='is_made_of_metal',
                                     item_standard='deve',
                                     req_pf=2
                                     ):
    items_all, df = pickle_wrap(get_standard_items_list,
                                kwargs={'pf_thresh': pf_thresh,
                                        'item_standard': item_standard},
                                easy_override=True)
    df = df[df['pf'] >= req_pf]
    items_w_feature = df[df['feature'] == feature]['concept'].unique()
    df_w = df[df['concept'].isin(items_w_feature) & (df['feature'] == feature)]
    pd.set_option('display.max_rows', None)
    print(df_w[['concept', 'pf']].sort_values('pf'))

    items_w_feature = df[df['feature'] == feature]['concept'].unique()
    items_without_feature = set(items_all) - set(items_w_feature)

    items_w_feature = sorted(list(items_w_feature))
    items_without_feature = sorted(list(items_without_feature))
    return items_w_feature, items_without_feature


def do_feature_LASSO(feature='is_for_women_females_girls',
                     quick=1):
    # feature = 'does_protect'
    items_w_feature, items_without_feature = (
        get_binary_feature_matrix_single(feature=feature))
    print(f'Number of items with feature ({feature} = 1): {len(items_w_feature)}')
    print(f'Number of items without feature ({feature} = 0): {len(items_without_feature)}')
    quit()
    y = np.array([1] * len(items_w_feature) + [0] * len(items_without_feature))
    items = items_w_feature + items_without_feature
    # items.sort()

    activations = get_item_vecs(items, quick=quick)
    print(f'{activations.shape=}')
    for layer in range(4, activations.shape[1]):
        test_regression_models(activations[:, layer, :], y, items)
        quit()


if __name__ == '__main__':
    do_feature_LASSO()
    # get_binary_feature_matrix_single()
