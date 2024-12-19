import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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

def norm_pf():
    # for each feature type, normalize the pf values based on
    #   how rare a given feature is across all items.
    #   I worry this may induce negative correlations but we'll see
    pass

def get_word_vecs(df):
    feat2total = df.groupby('feature')['pf'].sum()
    type2feat_list, type2feat_map = get_type2feat_list(df)
    type2feat_matrix = {}
    num_items = df['concept'].nunique()
    for feat_type, feat_list in type2feat_list.items():
        type2feat_matrix[feat_type] = np.zeros((num_items, len(feat_list)))

    for i, (item, df_item) in enumerate(df.groupby('concept')):
        for feature_type, feat_list in type2feat_list.items():
            vec = [0] * len(feat_list)
            feat_map = type2feat_map[feature_type]
            for feat, pf in zip(df_item['feature'], df_item['pf']):
                if feat in feat_map:
                    vec[feat_map[feat]] = pf / feat2total[feat]
            type2feat_matrix[feature_type][i, :] = vec

    for feat_type, mat in type2feat_matrix.items():

        RSM = np.corrcoef(mat)
        RSM[np.diag_indices_from(RSM)] = np.nan
        plt.imshow(RSM, aspect='auto', interpolation='none')
        plt.title(f'{feat_type=}')
        plt.colorbar()
        plt.show()
        # print(RSM.shape)
    quit()


def get_devereux_RSM_by_type():
    fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    df = pd.read_csv(fp)
    feature_types = df['feature type'].unique()
    type2RSM = {}
    items = df['concept'].unique()
    df = fix_feature_type_classification(df)
    get_word_vecs(df)


    # df = fix_feature_type_classification(df)

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

            # intersect = set0.intersection(set1)
            # print(f'{type0=}, {type1=}, {intersect=}')
            assert len(set0.intersection(set1)) == 0
    quit()

    for item, df_item in df.groupby('concept'):
        pass


    for feature_type in feature_types:
        df_type = df[df['feature type'] == feature_type]


        # RSM = df_type.pivot(index='concept', columns='concept')
        # type2RSM[feature_type] = RSM
        # print(RSM)
        # quit()


if __name__ == '__main__':
    get_devereux_RSM_by_type()

    # fp = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    # df = pd.read_csv(fp)
    # pd.set_option('display.max_rows', None)
    # # print(df['feature'].value_counts())
    #
    # unique_items = df['concept'].nunique()
    # print(f'{unique_items=}')
    #
    # print(df['feature type'].value_counts())