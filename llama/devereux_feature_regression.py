from Utils.pickle_wrap_funcs import pickle_wrap
from llama.Rissman_similarity_analysis import fit_regularized_models
from llama.devereux_llama import get_llama_vecs_ar, get_standard_items_list
from llama.devereux_neuron import get_binary_feat_matrix
import numpy as np
import pandas as pd
from scipy import stats, spatial
import matplotlib.pyplot as plt

from llama.devereux_w2v import get_w2v_deve_vecs


def find_closest_feature(feat2onehot):
    feats = list(feat2onehot.keys())
    print(len(feats))
    feat_vecs = np.array(list(feat2onehot.values()))
    dists = stats.spearmanr(feat_vecs.T).correlation
    feat2closest = {}
    feat2closest_corr = {}
    for i, feat in enumerate(feats):
        closest = np.argsort(dists[i])[-2]
        print(f'{feat} | {feats[closest]} | r = {dists[i, closest]:.2f}')
        feat2closest[feat] = feats[closest]
        feat2closest_corr[feat] = dists[i, closest]
    return feat2closest, feat2closest_corr


def examine_feature_reliability(pf_thresh=600, cat='gate_proj_in', layer_name=1,
                          activation_model='meta-llama/Llama-3.2-3b',
                          normalize=False, quick=1, item_standard='deve',
                          position=0, symmetric=False,):
    items, df = get_standard_items_list(pf_thresh, item_standard)

    feat2onehot0 = pickle_wrap(get_binary_feat_matrix,
                              kwargs={'items': items,
                                      'pf_thresh': pf_thresh,
                                      'threshold': 30,
                                      'item_std': item_standard,
                                      'req': 3,
                                      'odd_even': 0,
                                      },
                              verbose=-1, easy_override=False,
                              RAM_cache=True)
    feat2onehot1 = pickle_wrap(get_binary_feat_matrix,
                               kwargs={'items': items,
                                       'pf_thresh': pf_thresh,
                                       'threshold': 30,
                                       'item_std': item_standard,
                                       'req': 3,
                                       'odd_even': 1,
                                       },
                               verbose=-1, easy_override=False,
                               RAM_cache=True)
    for feat in feat2onehot0.keys():

        onehot0 = feat2onehot0[feat]
        onehot1 = feat2onehot1[feat]
        print(f'{feat} | {np.mean(onehot0):.2f} | {np.mean(onehot1):.2f} | '
              f'{stats.spearmanr(onehot0, onehot1).correlation:.2f}')
    quit()

def do_feature_regression(feature='is_small',
                          pf_thresh=300, cat='gate_proj_in', layer_name=1,
                          activation_model='meta-llama/Llama-3.2-3b',
                          normalize=False, quick=1, item_standard='deve',
                          position=0, symmetric=False,):
    # quick = 'scenes'
    # position = 1
    # activation_model = 'simCSE'

    vecs_by_layer = []
    # for layer_name in range(8, 20):
    vecs = get_llama_vecs_ar(pf_thresh, cat, layer_name,
                             activation_model, normalize, quick, item_standard,
                             position, symmetric)
    # RSM_0 = np.corrcoef(vecs)
    # vecs_by_layer.append(vecs)
    # vecs = np.nanmean(vecs_by_layer, axis=0)
    vecs_scn = get_llama_vecs_ar(pf_thresh, cat, layer_name,
                             activation_model, normalize, 'scenes', item_standard,
                             1, symmetric)
    # RSM_1 = np.corrcoef(vecs_scn)
    # trils = np.tril_indices_from(RSM_0, k=-1)
    # RSM_0 = RSM_0[trils]
    # RSM_1 = RSM_1[trils]
    # r, p = stats.spearmanr(RSM_0, RSM_1, nan_policy='omit')
    # print(f'{r=:.2f} {p=:.2f}')
    # quit()
    # vecs = np.hstack([vecs, vecs_scn])
    # print(vecs.shape)
    # # vecs = vecs_scn - vecs
    vecs = np.nanmean([vecs, vecs_scn], axis=0)


    # print(vecs.shape)

    # vecs = get_w2v_deve_vecs(pf_thresh, item_standard)

    items, df = get_standard_items_list(pf_thresh, item_standard)

    feature2type = df.groupby('feature')['feature type'].first().to_dict()

    feat2onehot = pickle_wrap(get_binary_feat_matrix,
                              kwargs={'items': items,
                                      'pf_thresh': pf_thresh,
                                      'threshold': 20,
                                      'item_std': item_standard,
                                      'req': 5,
                                      },
                              verbose=-1, easy_override=False,
                              RAM_cache=True)
    feat2closest, feat2closest_corr = find_closest_feature(feat2onehot)

    print(feat2onehot.keys())
    print(f'{len(feat2onehot)=}')

    feature_l = []
    feature_types_l = []
    scores = []
    for feature, onehot in feat2onehot.items():
        feature_type = feature2type[feature]
        res = fit_regularized_models(vecs, onehot)
        r2_score = res['Ridge']['r2_score']
        # r2_score = np.random.normal()
        print(f'{feature_type} | {feature} ({np.mean(onehot):.2f}) | '
              f'{r2_score=:.2f}')
        feature_l.append(feature)
        scores.append(r2_score)
        feature_types_l.append(feature_type)
    df_res = pd.DataFrame({'feature': feature_l,
                           'feature_type': feature_types_l,
                           'r2_score': scores})
    df_res['r'] = df_res['r2_score'] ** 0.5
    df_res['closest'] = df_res['feature'].map(feat2closest)
    feat2r = df_res.set_index('feature')['r'].to_dict()
    df_res['closest_r'] = df_res['closest'].map(feat2r)
    df_res['closest_corr'] = df_res['closest'].map(feat2closest_corr)
    df_res['closest_mediation'] = df_res['closest_corr'] * df_res['closest_r']

    df_res.sort_values('r2_score', ascending=False, inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    df_res['feature'] = df_res['feature'].apply(lambda x: x[:15])
    df_res['closest'] = df_res['closest'].apply(lambda x: x[:15])
    df_res['dif'] = df_res['r'] - df_res['closest_mediation']

    for key in ['r', 'closest_r', 'closest_corr', 'closest_mediation',
                'dif']:
        df_res[key] = df_res[key].apply(lambda x: f'{x:.2f}')

    print(df_res[['feature_type', 'feature', 'r', 'closest',
                  'dif', 'closest_mediation',
                  'closest_r', 'closest_corr',
                  ]])





if __name__ == '__main__':
    # examine_feature_reliability()
    do_feature_regression(layer_name=8)





