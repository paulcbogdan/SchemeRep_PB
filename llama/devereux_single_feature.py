import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from llama.Rissman_similarity_analysis import fit_regularized_models
from llama.devereux_feature_regression import get_vecs_for_regr
from llama.devereux_llama import get_standard_items_list
from llama.devereux_neuron import get_binary_feat_matrix
from marinate.pkld import pkld


@pkld
def regression_one_feature(feature='is_small',
                           pf_thresh=300,
                           cat='gate_proj_in', layer_name=8,
                           activation_model='meta-llama/Llama-3.2-3b',
                           # activation_model='meta-llama/Llama-3.3-70b-Instruct',
                           normalize=False, quick=1, item_standard='deve',
                           position=0, symmetric=False,
                           req=5, normalize_regr=False,
                           do_r2=True
                           ):
    vecs = get_vecs_for_regr(activation_model, pf_thresh, item_standard, cat,
                             layer_name, normalize, quick, position, symmetric,
                             store_='both'
                             )

    # items, df = get_standard_items_list(pf_thresh, item_standard)

    items, _ = get_standard_items_list(pf_thresh, item_standard)
    items = items[:pf_thresh]
    # print(len(items))
    # quit()
    feat2onehot = get_binary_feat_matrix(items, item_standard, threshold=10,
                                         pf_thresh=300, req=req)
    d_add = {}
    for feat, onehot in feat2onehot.items():
        d_add[feat[:15]] = onehot
    feat2onehot = {**feat2onehot, **d_add}

    try:
        onehot = feat2onehot[feature]
    except KeyError:
        print(f'{feature=} not found in {feat2onehot.keys()}')
        raise KeyError
    res = fit_regularized_models(vecs, onehot, normalize=normalize_regr,
                                 do_r2=do_r2)
    return res['Ridge']['r2_score']


def plot_all_feats(#cat='input',
                   #cat='down_proj_out',
                   cat='attn_output',
                   #  cat='gate_proj_in',
                   # activation_model='meta-llama/Llama-3.3-70b-Instruct',
                   pf_thresh=300, threshold=20, req=5,
                   activation_model='meta-llama/Llama-3.2-3b',
                   ):
    # feats = get_final_feats()#[::-1]

    items, _ = get_standard_items_list(pf_thresh, 'deve')
    feat2onehot = get_binary_feat_matrix(items, 'deve', threshold=20,
                                         pf_thresh=pf_thresh, req=req)
    feats_og = list(feat2onehot)
    r2s_all = []
    feats = ['is_a_mammal', 'is_a_bird', 'is_worn', 'is_a_musical_in', 'has_feathers', 'is_an_insect', 'has_wings',
             'is_clothing', 'has_wheels', 'is_a_vegetable', 'is_eaten_edible', 'is_a_fruit', 'made_of_metal',
             'is_found_in_sea', 'made_of_fabric_', 'has_skin_peel', 'is_a_weapon', 'is_a_plant', 'has_fur_hair',
             'does_fly', 'does_swim', 'is_electric', 'is_found_in_kit', 'is_an_animal', 'is_dangerous',
             'does_make_sound', 'has_a_handle_ha', 'does_carry_tran', 'made_of_wood', 'has_legs', 'is_pretty_attra',
             'is_a_tool', 'is_warm', 'has_a_tail', 'has_teeth', 'is_food', 'is_big_large', 'is_for_children',
             'made_of_plastic', 'has_claws', 'is_expensive', 'is_circular_rou', 'does_smell_is_s', 'is_colourful',
             'is_thin', 'is_small', 'is_fast', 'made_of_glass', 'is_heavy', 'is_soft', 'is_strong']
    feats = feats[:50]
    feats = [feat for feat in feats if feat in feats_og]
    feats = feats[:20]
    print(f'{len(feats)=}')

    for feat in feats:  # [:5]:
        r2s = []
        for layer_name in tqdm(range(80 if '70b' in activation_model else 28)):
            r2 = regression_one_feature(feat, cat=cat,
                                        layer_name=layer_name,
                                        activation_model=activation_model,
                                        # do_r2='5050',
                                        do_r2=False,
                                        pf_thresh=pf_thresh, req=req)
            r2s.append(r2)
            # except KeyError:
            #     print(f'Failed for {feat=}, {layer_name=}')
            #     break
        else:
            # r2s = np.array(r2s)
            r2s_std = stats.zscore(r2s)
            # if r2s_std[0] > -2: continue
            r2s -= r2s[0]
            r2s_all.append(r2s)

            plt.plot(r2s_std, label=feat, alpha=0.5,
                     marker='o', markersize=2)
            # plt.title(feat)
            # plt.show()
            # print(F'{feat}: {r2s[0]:.2f} {r2s[-1]:.2f}')
    # # plt.legend()
    plt.show()
    quit()
    r2s_all = np.array(r2s_all)
    # r2s_all -= np.nanmean(r2s_all, axis=1)[:, None]
    # r2s_all -= r2s_all[:, 0][:, None]
    r2s_M = np.nanmean(r2s_all, axis=0)
    print(f'{r2s_all.shape=}')

    confidence_interval = np.std(r2s_all, axis=0) / np.sqrt(r2s_all.shape[0])

    plt.plot(r2s_M, alpha=0.5,
             marker='o', markersize=2,
             color='blue')
    plt.fill_between(list(range(r2s_M.shape[0])),
                     r2s_M - confidence_interval,
                     r2s_M + confidence_interval,
                     color='dodgerblue', alpha=0.2,
                     # label='Confidence Interval'
                     )
    plt.title(f'{cat=}, {activation_model=}')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_all_feats()
