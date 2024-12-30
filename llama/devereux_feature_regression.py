from Utils.pickle_wrap_funcs import pickle_wrap
from llama.Rissman_similarity_analysis import fit_regularized_models
from llama.devereux_llama import get_llama_vecs_ar, get_standard_items_list
from llama.devereux_neuron import get_binary_feat_matrix
import numpy as np
import pandas as pd
from scipy import stats, spatial
import matplotlib.pyplot as plt

from llama.devereux_w2v import get_w2v_deve_vecs
from marinate import marinate


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

@marinate(overwrite=False)
def do_feature_regression(feature='is_small',
                          pf_thresh=600, cat='gate_proj_in', layer_name=8,
                          activation_model='meta-llama/Llama-3.2-3b',
                          normalize=False, quick=1, item_standard='deve',
                          position=0, symmetric=False,):

    if activation_model == 'w2v':
        vecs = get_w2v_deve_vecs(pf_thresh, item_standard)
    else:
        if isinstance(layer_name, list):
            vecs = []
            for layer in layer_name:
                vecs_layer = get_llama_vecs_ar(pf_thresh, cat, layer,
                                         activation_model, normalize,
                                         quick, item_standard,
                                         position, symmetric)
                vecs.append(vecs_layer)
            vecs = np.concatenate(vecs, axis=1)
        else:
            vecs = get_llama_vecs_ar(pf_thresh, cat, layer_name,
                                     activation_model, normalize,
                                     quick, item_standard,
                                     position, symmetric)

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

    bad_feats = ['has_a_beak', 'does_grow',  # 'has_wings',
                 'has_skin_peel'  # basically fruit?
                 'has_feathers'  # basically bird?
                 'made_of_fabric_cloth_material',  # maybe
                 'is_sweet', 'has_leaves', 'is_tasty',
                 'has_fur', 'does_make_sound', 'has_roots',
                 'has_an_engine',
                 'is_found_in_kitchen',
                 'has_a_blade', 'made_of_cotton', 'does_eat',
                 'has_a_stalk_stem',
                 'is_juicy',
                 'does_kill', 'has_four_legs', 'has_skin',
                 'is_healthy', 'has_a_seat_seats',
                 'is_useful',  # similar to "is_a_tool"
                 'is_used_in_cooking',
                 'is_long',
                 'is_green',
                 'does_protect',  # similar to clothing
                 # 'is_pretty_attractive'
                 'has_flesh',
                 'does_live_in_water', 'is_yellow',
                 'is_black', 'is_red', 'is_brown', 'is_heavy',
                 'does_lay_eggs', 'has_eyes', 'is_hard',
                 'is_grown', 'is_colorful', 'is_white', 'is_pink',
                 'is_strong'
                 ]

    feature_l = []
    feature_types_l = []
    scores = []
    for feature, onehot in feat2onehot.items():
        if feature in bad_feats: continue
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
    df_res['r2_score'] = df_res['r2_score'].fillna(0)
    df_res['r'] = df_res['r'].fillna(0)
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
    return df_res

def plot_feat_regr():
    model_name = ['simCSE']
    # df_res = do_feature_regression(activation_model='simCSE')
    df_w2v = do_feature_regression(activation_model='w2v')
    df_w2v.loc[df_w2v['r'].astype(float) < 0] = 0
    df_llama = do_feature_regression()#layer_name=list(range(8, 16)))
    df_llama.loc[df_llama['r'].astype(float) < 0] = 0
    df_llama = df_llama.iloc[:50]
    df_llama = df_llama.iloc[::-1]
    x_llm = df_llama['feature']
    y_llm = np.array(df_llama['r'].astype(float).to_numpy() ** 2)

    d_llama = {x: y for x, y in zip(x_llm, y_llm)}
    d_w2v = {x: y for x, y in zip(df_w2v['feature'],
                                  df_w2v['r'].astype(float).to_numpy() ** 2)}
    y_w2v = np.array([d_w2v.get(x, 0) for x in x_llm])
    y_dif = y_llm - y_w2v


    # print(y_llm)
    plt.figure(figsize=(4, 6.5))
    plt.xlim(0, 1)
    plt.barh(x_llm, y_w2v, color='gray')
    y_dif[y_dif < 0] = 0
    plt.barh(x_llm, y_dif, left=y_w2v, color='red')
    for i in range(len(x_llm)):
        if y_w2v[i] > y_llm[i] - 0.001:
            plt.plot([y_llm[i], y_llm[i]],
                     [i - 0.3, i + 0.35], 'r-',
                     linewidth=1.5)
    # yt = np.arange(len(x_llm))
    # print(yt)
    # plt.barh(yt , y_w2v, 0.4, color='gray')
    # plt.barh(yt + 0.5, y_llm, 0.4, color='dodgerblue')
    # plt.yticks(yt, x_llm)
    plt.ylim(-0.75, 49.75)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    print(df_llama)




if __name__ == '__main__':
    plot_feat_regr()
    # examine_feature_reliability()
    # do_feature_regression(layer_name=8)





