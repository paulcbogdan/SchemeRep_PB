from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

def convert_mariam2deve():
    fp_mariam = r'C:\PycharmProjects\SchemeRep\llama\features\Mariam_raw.csv'
    df_mariam = pd.read_csv(fp_mariam)
    concept_feature2cnt = defaultdict(int)
    concept2features = defaultdict(set)
    concept_feature2feature_type = {}
    concepts = set()
    cnt_errors = 0
    for i, row in df_mariam.iterrows():
        concept = row['Concept']
        feature = row['Feature']
        feature_type = row['Label']
        concept_feature2cnt[(concept, feature)] += 1
        if feature_type in concept_feature2feature_type:
            cnt_errors += 1
        concept_feature2feature_type[(concept, feature)] = feature_type
        concepts.add(concept)
        concept2features[concept].add(feature)
    print(f'Number of different labeled feature types: {cnt_errors}')

    concepts = sorted(list(concepts))
    df_as_l = []
    for concept in concepts:
        for feature in concept2features[concept]:
            pf = concept_feature2cnt[(concept, feature)]
            feature_type = concept_feature2feature_type[(concept, feature)]
            d = {'concept': concept, 'feature': feature, 'pf': pf,
                 'feature type': feature_type}
            df_as_l.append(d)
    df_out = pd.DataFrame(df_as_l)

    fp_deve = r'C:\PycharmProjects\SchemeRep\llama\features\Devereux_norm_dict.csv'
    df_deve = pd.read_csv(fp_deve)
    items_dev = set(df_deve['concept'])
    df_out['in_both'] = df_out['concept'].apply(lambda x: x in items_dev)

    items_mariam = set(df_out['concept'])
    df_deve['in_both'] = df_deve['concept'].apply(lambda x: x in items_mariam)
    df_deve.to_csv(fp_deve, index=False)

    fp_out = r'C:\PycharmProjects\SchemeRep\llama\features\Mariam_norm_dict.csv'
    df_out.to_csv(fp_out, index=False)

    item2pf = df_out.groupby('concept')['pf'].sum()
    pfs = item2pf.values
    plt.hist(pfs, bins=40)
    plt.show()
    pd.set_option('display.max_rows', None)
    print(item2pf.sort_values())

if __name__ == '__main__':
    convert_mariam2deve()