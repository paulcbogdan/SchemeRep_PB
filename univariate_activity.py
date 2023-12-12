import itertools
from collections import defaultdict

from atlas_utils import get_atlas
from conn_utils import get_BNA_ROIs
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_all_sns, get_trial_info
import numpy as np
import scipy.stats as stats

from utils import pickle_wrap
from collections.abc import Iterable as iterable
import pandas as pd
from rpy2 import rinterface
from pandas.api.types import is_numeric_dtype

def prep_activation_for_univariate(fp='obj3_fMRI', key='inc', conds=(1, 3),
                                   keys=('inc', 'con_hit'),
                                   # conds=((1, 3), (False, True))
                                   ):
    atlas = get_atlas(combine_regions=False, combine_bilateral=False,
                      shenyang=True)
    coords = atlas['coords']
    age2sn = get_all_sns(ret=True)
    ROI2age2inc2l = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    df_l = []
    ROI_l = None
    ROI_cols = None
    for i, age in enumerate([1, 2]):
        sns = age2sn[age]
        # sns = ['105']
        for sn in sns:
            print(f'Prepping univariate: {sn=}')
            df_sn = get_trial_info(sn, easy_override=True)
            ROI2vecs0 = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=False,
                                     easy_override=False,
                                     combine_regions=False)
            if ROI_l is None:
                ROI_l = list(ROI2vecs0.keys())

            ar = []
            for ROI in ROI_l:
                activity = ROI2vecs0[ROI]
                ar.append(np.nanmean(activity, axis=1))
                ROI_cols = ['_'.join(x.split()[1:]) for x in ROI_l]
            df_sn_ROI = pd.DataFrame(np.array(ar).T, columns=ROI_cols)
            df_sn = pd.concat([df_sn, df_sn_ROI], axis=1)
            df_sn['age'] = age
            df_l.append(df_sn)
    return pd.concat(df_l), ROI_cols


def dd_to_d(dd):
    if isinstance(dd, iterable):
        if isinstance(dd, list):
            return np.array(dd)
        elif isinstance(dd, dict) or isinstance(dd, defaultdict):
            plain_dict = {}
            for k, v in dd.items():
                plain_dict[k] = dd_to_d(v)
            return plain_dict
        else:
            raise TypeError(f'Unknown iterable type: {type(dd)}')
    return dd


def keys_from_formula(formula):
    formula = formula.split('~')[0]
    formula = formula.split('+')
    formula = [x.strip() for x in formula]
    return formula

def mean_or_fist(l):
    if is_numeric_dtype(l):
        return l.mean()
    else:
        return l.iloc[0]

def get_shenyang_subjects():
    sns_str = '102 103 105 106 107 108 110 111 112 114 116 117 120 123 124 125 ' \
              '126 127 128 130 132 134 135 136 137 201 202 203 ' \
              '205 206 207 208 210 211 212 214 216 217 218 219 221 222 225 230 233 234'
    return set(sns_str.split())


def do_univariate_analysis(fp='obj3_fMRI'):
    kwargs = {'fp': fp, 'key': 'inc', 'conds': (1, 3)}
    df, ROI_cols = pickle_wrap(None, prep_activation_for_univariate,
                                kwargs=kwargs, cache_dir='cache',
                                easy_override=False)
    df = df[~pd.isna(df['per_inc14_str'])]
    df = df[df['con_hit'] > 0]

    df_test = df[df['sn'] == '104']
    print(df_test['con_hit'])

    sns_sh = get_shenyang_subjects()
    sns_mine = set(df['sn'].unique())
    for sn in sns_sh:
        if sn not in sns_mine:
            print(f'shenyang\'s {sn} not in mine')
    print('-')
    for sn in sns_mine:
        if sn not in sns_sh:
            print(f'my {sn} not in shenyang\'s')
    quit()

    df = df[df['sn'].isin(sns_sh)]

    # df = df.groupby(['age', 'sn', 'inc_str', 'con_hit']).agg(mean_or_fist).\
    #     reset_index()
    # df = df.groupby(['age', 'sn', 'inc_str', 'con_hit']).mean().reset_index()
    # df = df[df['inc_str'] != 'neutral']

    sns = df['sn'].unique()
    print(len(sns))
    # sns_str = ' '.join(sns)
    # print(sns_str)
    quit()

    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    # print(df)
    # print(list(df.columns))
    # print(df['obj'])
    # quit()
    # print(len(df))
    # quit()

    # 102 103 105 106 107 108 110 111 112 114 116 117 120 123 124 125 126 127 128 130 132 134 135 136 137
    # 102 103 104 105 106 107 108 109 110 111 112 113 114 115 117 118 119 120 123 124 126 127 128 129 130 131 132 134 136 137 201 202 203 204 205 206 207 208 209 210 211 214 216 217 218 219 221 222 225 227 230 232 233 235

    from pymer4.models import Lmer
    for ROI in ROI_cols:
        print(f'ROI: {ROI}')
        if ROI != 'ATL_R_6_5':
            continue
        # formula = f'{ROI} ~ age*per_inc14_str*con_hit + (1|sn) + (1|obj)'
        formula =  f'{ROI} ~ age + (1|sn) + (1|obj)'
        # print(f'Formula: {formula}')

        keys = keys_from_formula(formula)
        # print(f'Keys: {keys}')

        model = Lmer(formula, data=df)
        try:
            model.fit(REML=True, verbose=True, summary=True)
        except Exception as e:
            print(f'Error fitting model: {e}')
            continue
        quit()

        summary = model.coefs
        print(summary.iloc[1:])
        t = summary['T-stat'].loc['(Intercept)']
        p = summary['P-val'].loc['(Intercept)']
        # print(model.fit())
        # quit()


if __name__ == '__main__':
    # prep_activation_for_univariate('bl3_fMRI')
    do_univariate_analysis()

