from collections import defaultdict
from functools import cache
import numpy as np
from Utils.pickle_wrap_funcs import pickle_wrap
import pandas as pd
from llama.devereux_llama import get_standard_items_list, get_deve_llama_RSM, prep_all_llama_d_vecs_deve, \
    get_dev_explore_BERT, ITEM_STANDARD

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
                    df.loc[(df['feature'] == feature) & (df['feature type'] == type1), 'feature type'] = type0
                else:
                    df.loc[(df['feature'] == feature) & (df['feature type'] == type0), 'feature type'] = type1
    return df


def get_type2feat_list(df):
    type2feat_list = {}
    type2feat_map = defaultdict(dict)
    for feat_type, df_feat_type in df.groupby('feature type'):
        type2feat_list[feat_type] = sorted(df_feat_type['feature'].unique().tolist())
        for i, feat in enumerate(type2feat_list[feat_type]):
            type2feat_map[feat_type][feat] = i
    return type2feat_list, type2feat_map



def get_type2RSM_(df, plot=False, attn=False, pf_thresh=250):
    feat2total = df.groupby('feature')['pf'].sum()
    type2feat_list, type2feat_map = get_type2feat_list(df)
    type2feat_matrix = {}
    num_items = df['concept'].nunique()
    for feat_type, feat_list in type2feat_list.items():
        type2feat_matrix[feat_type] = np.zeros((num_items, len(feat_list)))

    standard_item_list, _ = get_standard_items_list(pf_thresh=pf_thresh)
    df_items = df['concept'].unique()
    assert set(standard_item_list) == set(df_items)
    df_grp = df.groupby('concept')
    for i, item in enumerate(standard_item_list):
        df_item = df_grp.get_group(item)
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
        RSM = np.corrcoef(mat)
        print(f'Cooked dev {feat_type}: {RSM.shape=}')
        RSM[np.diag_indices_from(RSM)] = np.nan
        type2RSM[feat_type] = RSM
        if plot:
            plt.imshow(RSM, aspect='auto', interpolation='none')
            plt.title(f'{feat_type=}')
            plt.colorbar()
            plt.show()
    return type2RSM

def get_devereux_RSM_by_type_(pf_thresh=250, attn=False, odd_even=None):
    items, df = get_standard_items_list(pf_thresh=pf_thresh)
    if odd_even is not None:
        assert odd_even in [0, 1]
        all_keep_features = []
        for feature_type, df_feat_type in df.groupby('feature type'):
            features = df_feat_type['feature'].value_counts()
            features_odd = features.index.to_list()[odd_even::2]
            all_keep_features.extend(features_odd)
        all_keep_features = set(all_keep_features)
        df = df[df['feature'].isin(all_keep_features)]

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
                   pf_thresh=300, quick=None, item_standard='deve'):
    models = get_explore_llama(activation_model=activation_model,
                               attn=False, st=0)
    # print(models)
    # quit()

    if 'llama' in models[0]:
        layers = 28
    else:
        layers = 13

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(1, 1, figsize=(6, 5.5))

    type2list = defaultdict(list)

    for layer in range(0, layers):
        model = models[layer]
        print(f'Layer ({model[1]}): {layer}')
        # pf_thresh = 300

        RSM = get_deve_llama_RSM(model, pf_thresh=pf_thresh,
                                 quick=quick, item_standard=item_standard,
                                 # position=None,
                                 )
        # RSM_no_context = get_deve_llama_RSM(model, pf_thresh=pf_thresh,
        #                                     quick=1, item_standard=item_standard)
        # RSM = RSM - RSM_no_context

        trils = np.tril_indices_from(RSM, k=-1)
        RSM_flat = RSM[trils]
        type2RSM = get_devereux_RSM_by_type(pf_thresh=pf_thresh,
            attn=isinstance(attn, bool) and attn, )


        for feature_type, RSM_feat in type2RSM.items():
            assert RSM_feat.shape == RSM.shape, f'{RSM_feat.shape=} {RSM.shape=}'
            RSM_feat_flat = RSM_feat[trils]
            RSM_feat_nans = np.isnan(RSM_feat_flat)
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

    print(type2list)
    for feat_type, l in type2list.items():
        plt.plot(l, label=feat_type, marker='.')
    # plt.xlim(0, 28)
    plt.ylim(0, .3)
    cat = models[0][1]
    quick = pf_thresh if quick is None else quick

    if 'llama' in models[0]:
        model_name = activation_model.split('/')[-1]
    else:
        model_name = models[0][1]

    title = (f'{model_name}\n'
             f'{pf_thresh} items, averaging across {quick} contexts')

    plt.title(title)
    fig.legend(ncol=2, frameon=False, loc='lower center')
    plt.xlabel('Layer')
    plt.ylabel('Spearman correlation (r)')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplots_adjust(bottom=0.32, left=.15, right=.95, top=.85)
    plt.show()





def examine_deve_dino_overlap():
    # test = get_standard_items_list(0)
    # print(len(test))
    # quit()

    concepts_std, _ = get_standard_items_list(pf_thresh=250)

    fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    df = pd.read_csv(fp)
    concepts = df['concept'].unique()

    fp = r'C:\PycharmProjects\SchemeRep\llama\features\ElectroDino_feature_matrix.csv'
    df_dino = pd.read_csv(fp)
    concepts_dino = df_dino['concept'].unique()
    print(f'{len(concepts)=}')
    print(f'{len(concepts_dino)=}')

    overlap = set(concepts).intersection(set(concepts_dino))
    not_dino = set(concepts) - set(concepts_dino)
    dino_only = set(concepts_dino) - set(concepts)
    print(f'{len(overlap)=}, {len(not_dino)=}, {len(dino_only)=}')

    dino_std = set(concepts_dino).intersection(set(concepts_std))
    print(f'{len(dino_std)=}')
    quit()



if __name__ == '__main__':
    do_llama_x_dev()

