from collections import defaultdict
from functools import cache
import numpy as np
from Utils.pickle_wrap_funcs import pickle_wrap
import pandas as pd
from llama.devereux_llama import get_standard_items_list, get_deve_llama_RSM, prep_all_llama_d_vecs_deve, \
    get_dev_explore_BERT, ITEM_STANDARD
from marinate import marinate
from marinate.pkld import pkld

try:
    import matplotlib.pyplot as plt


    from llama.model_settings import get_explore_llama
    from scipy import stats
except ModuleNotFoundError:
    pass

import warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')

# suppress RuntimeWarning: invalid value encountered in divide
np.seterr(divide='ignore', invalid='ignore')

def fix_feature_type_classification(df):
    type2features = {}
    type2feature_cnt = {}
    for feat_type, df_feat_type in df.groupby('feature type'):
        type2features[feat_type] = df_feat_type['feature'].unique()
        type2feature_cnt[feat_type] = df_feat_type['feature'].value_counts().to_dict()

    for type0 in type2features:
        for type1 in type2features:
            if type0 >= type1:
                continue
            set0 = set(type2features[type0])
            set1 = set(type2features[type1])
            intersect = set0.intersection(set1)
            for feature in intersect:
                cnt0 = type2feature_cnt[type0][feature]
                cnt1 = type2feature_cnt[type1][feature]
                if cnt0 > cnt1:
                    df.loc[(df['feature'] == feature) &
                           (df['feature type'] == type1), 'feature type'] = type0
                else:
                    df.loc[(df['feature'] == feature) &
                           (df['feature type'] == type0), 'feature type'] = type1
    return df


def get_type2feat_list(df):
    type2feat_list = {}
    type2feat_map = defaultdict(dict)
    for feat_type, df_feat_type in df.groupby('feature type'):
        type2feat_list[feat_type] = sorted(df_feat_type['feature'].unique().tolist())
        for i, feat in enumerate(type2feat_list[feat_type]):
            type2feat_map[feat_type][feat] = i
    return type2feat_list, type2feat_map



def get_type2RSM_(df, plot=False, attn=False, pf_thresh=250,
                  trial_sim='corr'):
    feat2total = df.groupby('feature')['pf'].sum()
    type2feat_list, type2feat_map = get_type2feat_list(df)
    type2feat_matrix = {}
    num_items = df['concept'].nunique()
    standard_item_list, _ = get_standard_items_list(pf_thresh=pf_thresh)
    for feat_type, feat_list in type2feat_list.items():
        type2feat_matrix[feat_type] = np.zeros((len(standard_item_list),
                                                len(feat_list)))

    df_items = df['concept'].unique()
    # assert set(standard_item_list) == set(df_items)
    df_grp = df.groupby('concept')
    for i, item in enumerate(standard_item_list):
        try:
            df_item = df_grp.get_group(item)
        except KeyError:
            for feature_type, feat_list in type2feat_list.items():
                type2feat_matrix[feature_type][i, :] = np.nan
            continue
        for feature_type, feat_list in type2feat_list.items():
            vec = [0] * len(feat_list)
            feat_map = type2feat_map[feature_type]
            # for each feature type, normalize the pf values based on
            #   how rare a given feature is across all items.
            #   I worry this may induce negative correlations but we'll see
            for feat, pf in zip(df_item['feature'], df_item['pf']):
                if feat in feat_map:
                    vec[feat_map[feat]] = pf / feat2total[feat]
            type2feat_matrix[feature_type][i, :] = vec

    print(f'Making devereux: {list(type2feat_matrix)}')
    type2RSM = {}
    for feat_type, mat in type2feat_matrix.items():
        if attn:
            mat_bool = mat > 0
            # weird af but at least this produces a symmetric histogram
            mat = stats.rankdata(mat_bool, axis=1, method='ordinal') # TODO: redo with just mat
            mat = stats.rankdata(mat, axis=0, method='ordinal')
            mat_std = stats.zscore(mat, axis=0, nan_policy='omit')
            mat_attn = []
            for i, item0 in enumerate(standard_item_list):
                for j, item1 in enumerate(standard_item_list):
                    if item0 >= item1:
                        continue
                    mat_vec = mat_std[i] * mat_std[j]
                    mat_attn.append(mat_vec)
            mat = np.array(mat_attn, dtype=np.int8)
        else:
            pass
        if trial_sim == 'corr':
            # plt.imshow(mat, aspect='auto', interpolation='none')
            # plt.show()
            # quit()
            # print(mat[0, :])
            from scipy import stats
            mat = stats.rankdata(mat, axis=1)
            # print(mat[0, :])
            # plt.imshow(mat, aspect='auto', interpolation='none')
            # plt.colorbar()
            # plt.show()
            # print(mat.shape)
            # quit()
            RSM = np.corrcoef(mat)
            RSM[np.diag_indices_from(RSM)] = np.nan
            # plt.imshow(RSM)
            # plt.colorbar()
            # plt.show()
            # quit()
        else:
            RSM = -np.abs(mat[:, None] - mat[None, :])
            RSM = np.mean(RSM, axis=-1)

        print(f'Cooked dev {feat_type}: {RSM.shape=}')
        RSM[np.diag_indices_from(RSM)] = np.nan
        type2RSM[feat_type] = RSM
        if plot:
            plt.imshow(RSM, aspect='auto', interpolation='none')
            plt.title(f'{feat_type=}')
            plt.colorbar()
            plt.show()
    return type2RSM

def get_devereux_RSM_by_type_(pf_thresh=250, attn=False, odd_even=None,
                              feat_min=5, all=False):
    items, df = get_standard_items_list(pf_thresh=pf_thresh)
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
    # df = df[df['feature'].isin(final_feats)]

    if feat_min is not None:
        feat_cnt = df['feature'].value_counts()
        df = df[df['feature'].isin(feat_cnt[feat_cnt >= feat_min].index)]
        num_features_by_type = df.groupby('feature type')['feature'].nunique()
        print(num_features_by_type)

    if all:
        df['feature type'] = 'all'

    features = list(df['feature'].unique())
    if odd_even is not None:
        features = set(features[odd_even::2])
        df = df[df['feature'].isin(features)]
        # assert odd_even in [0, 1]
        # all_keep_features = []
        # for feature_type, df_feat_type in df.groupby('feature type'):
        #     features = df_feat_type['feature'].value_counts()
        #     features_odd = features.index.to_list()[odd_even::2]
        #     all_keep_features.extend(features_odd)
        # all_keep_features = set(all_keep_features)
        # df = df[df['feature'].isin(all_keep_features)]


    df = fix_feature_type_classification(df)
    type2RSM = get_type2RSM_(df, attn=attn, pf_thresh=pf_thresh)
    return type2RSM

@cache
def get_devereux_RSM_by_type(pf_thresh=250, attn=False, odd_even=None):
    type2RSM = pickle_wrap(get_devereux_RSM_by_type_,
                           kwargs={'pf_thresh': pf_thresh, 'attn': attn,
                                   'odd_even': odd_even},
                           easy_override=True, verbose=-1)
    print('Got devereux RSMs')
    return type2RSM

@pkld(store='both', verbose=1, overwrite=True)
def get_devereux_top50_RSM(pf_thresh=300, odd_even=None):
    items = ['does_swim', 'is_a_mammal', 'is_clothing', 'is_worn', 'does_make_sound', 'has_a_tail', 'is_found_in_sea', 'is_a_musical_in', 'is_an_animal', 'is_eaten_edible', 'is_a_bird', 'is_a_plant', 'does_carry_tran', 'does_fly', 'has_claws', 'has_fur_hair', 'has_wings', 'is_an_insect', 'is_a_fruit', 'has_skin_peel', 'is_a_weapon', 'is_a_vegetable', 'has_wheels', 'has_teeth', 'is_fast', 'is_electric', 'has_legs', 'is_pretty_attra', 'has_feathers', 'made_of_fabric_', 'made_of_metal', 'is_warm', 'made_of_wood', 'made_of_glass', 'is_a_tool', 'is_found_in_kit', 'is_colourful', 'is_food', 'is_for_children', 'made_of_plastic', 'has_a_handle_ha', 'is_strong', 'is_dangerous', 'is_heavy', 'is_big_large', 'does_smell_is_s', 'is_circular_rou', 'is_small', 'is_expensive', 'is_soft', 'is_thin']
    print(len(items))
    top50 = set(items)
    items, df = get_standard_items_list(pf_thresh=pf_thresh, norm_per_concept=False)
    df = df[df['feature'].apply(lambda x: x[:15]).isin(top50)] # TODO: fix to be same as items
    features = list(df['feature'].unique())
    if odd_even is not None:
        features = set(features[odd_even::2])
        df = df[df['feature'].isin(features)]
    # TODO: clean this up when refactor. stop using the :15. Have items @pkld
    df['pf'] = df['pf'].apply(lambda x: 1 if x > 4 else 0)
    df = df[df['pf'] > 0]
    df['feature type'] = 'top50'
    type2RSM = get_type2RSM_(df, attn=False, pf_thresh=pf_thresh)
    if odd_even is not None:
        if odd_even == 1:
            type2RSM['Human (odd feats)'] = type2RSM['top50']
        else:
            type2RSM['Human (even feats)'] = type2RSM['top50']
    else:
        type2RSM['Human'] = type2RSM['top50']
    return type2RSM


def get_item_sum_RSM(model):
    RSM = get_deve_llama_RSM(model)
    items, _ = get_standard_items_list(pf_thresh=250)
    item2keys = {item: i for i, item in enumerate(items)}
    for i, item0 in enumerate(items):
        for j, item1 in enumerate(items):
            if item0 >= item1:
                continue
            idx0 = item2keys[item0]
            idx1 = item2keys[item1]
            RSM[i, j] = RSM[idx0, idx1]

def do_llama_x_dev(attn=False, activation_model='meta-llama/Llama-3.2-3b',
                   pf_thresh=300, quick=1, item_standard='deve',
                   position=0, one_line=True):

    if position == 0:
        quick = 1
    if 'llama' in activation_model.lower():
        models = get_explore_llama(activation_model=activation_model,
                                   attn=False, st=0, normalize=True)
    elif 'simCSE' in activation_model or 'BERT' in activation_model:
        models = get_dev_explore_BERT(activation_model)
    elif activation_model == 'w2v':
        return {'top50': [.478]}
        # val = .416 # claculated via devereux_w2v.py
        # label = 'word2vec'
        # plt.plot([14], [val], label=label, color='red',
        #          marker='o', linewidth=3, markersize=8)
        # plt.plot([13.1, 14.9], [val, val], color='red',
        #          linewidth=3)
        # return

    # models = get_dev_explore_BERT('simCSE')

    num_layers = len(models)
    # print(f'{num_layers=}')
    # quit()
    # models = get_explore_llama(activation_model=activation_model,
    #                            attn=False, st=7, end=21, normalize=True)
    # models = [models]

    type2list = defaultdict(list)

    for layer in range(0, num_layers):
        model = models[layer]
        print(f'Layer ({model[1]}): {layer}')
        # pf_thresh = 300

        RSM = get_deve_llama_RSM(model, pf_thresh=pf_thresh,
                                 quick=quick,
                                 item_standard=item_standard,
                                 position=position,
                                 symmetric=False
                                 )
        RSM[np.diag_indices_from(RSM)] = np.nan
        trils = np.tril_indices_from(RSM, k=-1)
        RSM_flat = RSM[trils]

        type2RSM = get_devereux_top50_RSM(pf_thresh=pf_thresh)
        if one_line:
            type2RSM_all = get_devereux_RSM_by_type(pf_thresh=pf_thresh,
                attn=isinstance(attn, bool) and attn, )
            type2RSM.update(type2RSM_all)

        for feature_type, RSM_feat in type2RSM.items():
            assert RSM_feat.shape == RSM.shape, f'{RSM_feat.shape=} {RSM.shape=}'
            RSM_feat_flat = RSM_feat[trils]
            RSM_feat_nans = np.isnan(RSM_feat_flat)
            RSM_llama_nanas = np.isnan(RSM_flat)
            RSM_feat_nans = RSM_feat_nans | RSM_llama_nanas
            num_nans = np.sum(RSM_feat_nans)
            RSM_flat_ = RSM_flat[~RSM_feat_nans]
            RSM_feat_flat_ = RSM_feat_flat[~RSM_feat_nans]
            r, p = stats.spearmanr(RSM_flat_, RSM_feat_flat_,
                                   nan_policy='omit' if num_nans > 0 else 'raise')

            prop_nan = num_nans / len(RSM_feat_flat)
            if prop_nan > 0:
                print(f'\t{feature_type=}: {r=:.3f}, {p=:.3f} ({prop_nan=:.1%})')
            else:
                print(f'\t{feature_type=}: {r=:.3f}, {p=:.3f}')
            type2list[feature_type].append(r)

    if not one_line:
        return type2list

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(1, 1, figsize=(6, 5.5))

    for feat_type, l in type2list.items():
        plt.plot(l, label=feat_type, marker='.')
    # plt.xlim(0, 28)
    plt.ylim(0, .7)
    cat = models[0][1]
    quick = pf_thresh if quick is None else quick

    if 'llama' in models[0]:
        model_name = activation_model.split('/')[-1]
    else:
        model_name = models[0][1]

    title = (f'{model_name}\n'
             f'{pf_thresh} items, averaging across {quick} contexts')
    title += f'\nPosition {position}'

    plt.title(title)
    fig.legend(ncol=2, frameon=False, loc='lower center')
    plt.xlabel('Layer')
    plt.ylabel('Spearman correlation (r)')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplots_adjust(bottom=0.32, left=.15, right=.95, top=.85)
    plt.show()





def examine_deve_dino_overlap():

    concepts_std, _ = get_standard_items_list(pf_thresh=250)

    fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    df = pd.read_csv(fp)
    concepts = df['concept'].unique()

    fp = r'C:\PycharmProjects\SchemeRep\llama\features\ElectroDino_feature_matrix.csv'
    df_dino = pd.read_csv(fp)
    concepts_dino = df_dino['concept'].unique()

    overlap = set(concepts).intersection(set(concepts_dino))
    not_dino = set(concepts) - set(concepts_dino)
    dino_only = set(concepts_dino) - set(concepts)
    print(f'{len(overlap)=}, {len(not_dino)=}, {len(dino_only)=}')

    dino_std = set(concepts_dino).intersection(set(concepts_std))
    quit()

def plot_all_deve_lines():
    models = ['meta-llama/Llama-3.2-3b', 'simCSE', 'BERT', 'w2v']
    colors = ['dodgerblue', 'orange', 'chocolate', 'red']
    for model, color in zip(models, colors):
        d = do_llama_x_dev(activation_model=model, one_line=False,
                           pf_thresh=300, quick=1)
        vals = d['top50']
        print(f'{vals=}')
        plt.plot(list(range(len(vals))), vals,
                 label=model, color=color,
                 marker='.')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_all_deve_lines()
    # do_llama_x_dev()

