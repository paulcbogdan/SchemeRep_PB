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
    feat_vecs = np.array(list(feat2onehot.values()))
    dists = stats.spearmanr(feat_vecs.T).correlation
    feat2closest = {}
    feat2closest_corr = {}
    for i, feat in enumerate(feats):
        closest = np.argsort(dists[i])[-2]
        # print(f'{feat} | {feats[closest]} | r = {dists[i, closest]:.2f}')
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

@marinate(overwrite=True)
def do_feature_regression(feature='is_small',
                          pf_thresh=600, cat='gate_proj_in', layer_name=8,
                          activation_model='meta-llama/Llama-3.2-3b',
                          normalize=False, quick=1, item_standard='deve',
                          position=0, symmetric=False,
                          req=10, normalize_regr=False):

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
                                      'req': req,
                                      },
                              verbose=-1, easy_override=False,
                              RAM_cache=True)
    # print(list(feat2onehot.keys()))

    if 'is_noisy_loud' in feat2onehot and 'does_make_sound_a_noise' in feat2onehot:
        feat2onehot['is_noisy'] = (feat2onehot['is_noisy_loud'] |
                                   feat2onehot['does_make_sound_a_noise'])
        feature2type['is_noisy'] = 'custom'
        del feat2onehot['is_noisy_loud']
        del feat2onehot['does_make_sound_a_noise']
    elif 'is_noisy_loud' in feat2onehot:
        feat2onehot['is_noisy'] = feat2onehot['is_noisy_loud']
        feature2type['is_noisy'] = 'custom'
        del feat2onehot['is_noisy_loud']
    elif 'does_make_sound_a_noise' in feat2onehot:
        feat2onehot['is_noisy'] = feat2onehot['does_make_sound_a_noise']
        feature2type['is_noisy'] = 'custom'
        del feat2onehot['does_make_sound_a_noise']

    feat2closest, feat2closest_corr = find_closest_feature(feat2onehot)


    bad_feats = ['has_a_beak', 'does_grow',  # 'has_wings',
                 'has skin_peel'  # basically fruit?
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
                 'is_noisy_loud', 'does make sound_a noise',
                 # 'is_pretty_attractive'
                 'has_flesh',
                 'does_live_in_water', 'is_yellow',
                 'is_black', 'is_red', 'is_brown', #'is_heavy',
                 'does_lay_eggs', 'has_eyes', 'is_hard',
                 'is_grown', 'is_colorful', 'is_white', 'is_pink',
                 # 'is_strong',
                 'has_a_blade_blades', 'has_a_point',
                 # 'does_smell_is_smelly',
                 'is_sharp',
                 'is_noisy_loud'
                 ]

    feature_l = []
    feature_types_l = []
    scores = []
    for feature, onehot in feat2onehot.items():
        if feature in bad_feats: continue
        feature_type = feature2type[feature]
        res = fit_regularized_models(vecs, onehot, normalize=normalize_regr)
        r2_score = res['Ridge']['r2_score']
        # r2_score = np.random.normal()
        print(f'{feature_type} | {feature} ({np.mean(onehot):.2f}) | '
              f'{r2_score=:.2f}')
        feature_l.append(feature)
        scores.append(r2_score)
        feature_types_l.append(feature_type)
    feat2onehot_ = {feat: feat2onehot[feat] for feat in feature_l}
    # feat2M = calculate_category_homogeneity(feat2onehot_)


    df_res = pd.DataFrame({'feature': feature_l,
                           'feature_type': feature_types_l,
                           'r2_score': scores})
    # df_res['feat_homogeneity'] = df_res['feature'].map(feat2M)
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

def calculate_category_homogeneity(feat2onehot):
    items, df = get_standard_items_list(600, 'deve')
    feat2onehot = pickle_wrap(get_binary_feat_matrix,
                              kwargs={'items': items,
                                      'pf_thresh': 600,
                                      'threshold': 20,
                                      'item_std': 'deve',
                                      'req': 5,
                                      },
                              verbose=-1, easy_override=False,
                              RAM_cache=True)
    feat2onehot = {feat[:15]: feat2onehot[feat] for feat in feat2onehot.keys()}
    feat2onehot = {feat: feat2onehot[feat] for feat in get_final_feats()}
    ar = np.array(list(feat2onehot.values()))
    dists = 1 - np.ma.corrcoef(ar.T)
    feat2M = {}
    for feat, onehot in feat2onehot.items():
        M_dist = np.mean(dists[onehot == 1, :][:, onehot == 1])
        feat2M[feat] = M_dist
    for feat in get_final_feats():
        print(f'{feat} | {feat2M[feat]:.2f}')
    quit()
    return feat2M

def plot_feat_regr(req=5, normalize_regr=False):
    model_name = ['simCSE']
    # df_w2v = do_feature_regression(activation_model='simCSE', layer_name=12)
    df_w2v = do_feature_regression(activation_model='w2v', req=req,
                                   normalize_regr=normalize_regr)
    df_w2v.loc[df_w2v['r'].astype(float) < 0] = 0
    df_w2v['feature'] = df_w2v['feature'].str.replace('_', ' ')

    df_llama = do_feature_regression(layer_name=list(range(4, 16)), req=req,
                                     normalize_regr=normalize_regr)
    df_llama.loc[df_llama['r'].astype(float) < 0] = 0

    # df_w2v = df_w2v.iloc[:50]
    # df_w2v = df_w2v.iloc[::-1]
    # df_llama = df_llama.iloc[:50]
    # df_llama = df_llama.iloc[::-1]


    x_llm = df_llama['feature'].str.replace('_', ' ').to_numpy()




    y_llm = np.array(df_llama['r'].astype(float).to_numpy() ** 2)

    # d_llama = {x: y for x, y in zip(x_llm, y_llm)}
    d_w2v = {x: y for x, y in zip(df_w2v['feature'],
                                  df_w2v['r'].astype(float).to_numpy() ** 2)}
    y_w2v = np.array([d_w2v.get(x, 0) for x in x_llm])

    mapper = {'is pretty attra': 'is attractive',
              'is big large': 'is large',
              'does smell is smelly': 'is smelly',
              'does carry tran': 'can carry things',
              'has fur hair': 'has fur',
              'is found in kit': 'found in kitchen',
              'is foudn in sea': 'found in sea',
              'is a musical in': 'is musical instrument',
              'is worn': 'can be worn',
              'has a handle ha': 'has a handle',
              'is eaten edible': 'is edible',
              'has skin peel': 'has peelable skin',
              # 'does make sound': 'makes sound',
              'is circular rou': 'is circular',
              'is food': 'is human food'
              }
    # is_edible is just "living"
    x_llm = [mapper.get(x, x) for x in x_llm]


    y_dif_llm = y_llm - y_w2v
    y_dif_llm[y_dif_llm < 0] = 0
    y_dif_w2v = y_w2v - y_llm
    y_dif_w2v[y_dif_w2v < 0] = 0
    y_gray = np.min(np.array([y_llm, y_w2v]), axis=0)

    plt.figure(figsize=(4, 6.5))
    y_dif_w2v[y_dif_w2v < 0] = 0

    plt.xlim(0, 1)
    plt.barh(x_llm, y_gray, color='gray')
    plt.barh(x_llm, y_dif_llm, left=y_gray, color='dodgerblue')
    plt.barh(x_llm, y_dif_w2v, left=y_gray, color='red')
    # plt.ylim(-0.75, 49.75)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.])#, rotation=90)
    plt.yticks(fontsize=9)
    plt.gca().spines[['right', 'bottom', ]].set_visible(False)
    plt.gca().tick_params(top=True, labeltop=True,
                          bottom=False, labelbottom=False)
    plt.xlabel('R² (binary prediction)', fontsize=11, labelpad=5)
    plt.gca().xaxis.set_label_position('top')

    plt.tight_layout()
    plt.show()
    pd.set_option('display.max_rows', None)
    print(df_llama[['feature_type', 'feature', 'r']])

def get_model_R2(name, layer):
    features = get_final_feats()
    df_w2v = do_feature_regression(activation_model=name, layer_name=layer)
    df_w2v = df_w2v[df_w2v['feature'].isin(features)]
    df_w2v.loc[df_w2v['r'].astype(float) < 0] = 0
    df_w2v['r2'] = df_w2v['r'].astype(float) ** 2
    M_r2 = df_w2v['r2'].mean()
    return M_r2


def plot_regr_lines():
    models = ['meta-llama/Llama-3.2-3b', 'simCSE', 'BERT', 'w2v']
    model2layers = {'meta-llama/Llama-3.2-3b': 28,
                    'simCSE': 13,
                    'BERT': 13,
                    'w2v': 1}
    colors = ['dodgerblue', 'orange', 'chocolate', 'red']
    for model, color in zip(models, colors):
        vals = []
        for layer in range(model2layers[model]):
            M_r2 = get_model_R2(model, layer)
            vals.append(M_r2)
        plt.plot(list(range(len(vals))), vals,
                 label=model, color=color,
                 marker='.')
        if 'llama' in model:
            for layer in range(len(vals)):
                plt.text(layer, vals[layer], str(layer), color=color,
                         ha='left', va='top')
    plt.legend()
    big_r = get_model_R2('meta-llama/Llama-3.2-3b', layer=list(range(4, 28)))
    plt.title(f'Big R²: {big_r:.2f} layers [4, 28)')
    plt.show()




def get_final_feats():
    final_feats = ['is_heavy', 'is_soft', 'is_thin', 'is_strong', 'is_colourful',
                   'is_fast', 'has_claws', 'made_of_glass', 'does_smell_is_s',
                   'is_circular_rou', 'is_big_large', 'is_for_children',
                   'has_teeth', 'made_of_plastic', 'made_of_wood',
                   'is_pretty_attra', 'is_noisy', 'has_a_tail', 'is_food',
                   'is_a_tool', 'is_found_in_kit', 'has_legs', 'is_dangerous',
                   'does_carry_tran', 'is_warm', 'is_electric', 'is_a_plant',
                   'has_a_handle_ha', 'does_make_sound', 'is_an_animal',
                   'is_found_in_sea', 'does_swim', 'is_a_fruit', 'made_of_fabric_',
                   'is_eaten_edible', 'has_fur_hair', 'made_of_metal', 'does_fly',
                   'is_a_weapon', 'is_clothing', 'is_an_insect', 'has_wheels',
                   'has_skin_peel', 'is_a_vegetable', 'has_wings', 'is_a_musical_in',
                   'is_worn', 'has_feathers', 'is_a_mammal', 'is_a_bird']
    return final_feats

if __name__ == '__main__':
    # plot_regr_lines()
    plot_feat_regr()
    # examine_feature_reliability()
    # do_feature_regression(layer_name=8)





