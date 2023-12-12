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
            # ROI_l = ROI_l[:1]
            # print(ROI_l)

            ar = []
            for ROI in ROI_l:
                # if ROI != '103 FuG_L_3_1':
                #     continue
                activity = ROI2vecs0[ROI]
                # print(f'{activity.shape=}')
                # test_M =  np.nanmean(activity, axis=1)
                # print(test_M)
                # print(f'{np.nanmean(test_M)=}')
                # quit()
                ar.append(np.nanmean(activity, axis=1))
                # print(ROI_l)
                # quit()
                ROI_cols = ['_'.join(x.split()[1:]) for x in ROI_l]
            # print(ROI_cols)
            # quit()
            df_sn_ROI = pd.DataFrame(np.array(ar).T, columns=ROI_cols)
            df_sn = pd.concat([df_sn, df_sn_ROI], axis=1)

            df_sn['age'] = age
            df_l.append(df_sn)



            # for cond in conds:
            #     matches = df_sn[key] == cond
            #     for ROI, activity in ROI2vecs0.items():
            #         ROI2age2inc2l[ROI][age][cond].append(
            #             np.nanmean(activity[matches]))
        # pairs = itertools.combinations(conds, 2)
        # for cond0, cond1 in pairs:
        #     print(f'Comparing {cond0=} {cond1=}')
        #     for ROI, inc2l in ROI2inc2l.items():
        #         l0 = inc2l[cond0]
        #         l1 = inc2l[cond1]
        #         t, p = stats.ttest_rel(l0, l1)
        #         print(f'{ROI=}: {t=:.3f}, {p=:.3f}')
        #         continue
        #     continue
    return pd.concat(df_l), ROI_cols
    # ROI2age2inc2l = dd_to_d(ROI2age2inc2l)
    # return ROI2age2inc2l

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

def do_univariate_analysis(fp='obj3_fMRI'):
    kwargs = {'fp': fp, 'key': 'inc', 'conds': (1, 3)}
    df, ROI_cols = pickle_wrap(None, prep_activation_for_univariate,
                                kwargs=kwargs, cache_dir='cache',
                                easy_override=False)
    df = df.groupby(['age', 'sn', 'inc_str', 'con_hit']).agg(mean_or_fist).\
        reset_index()
    # df = df.groupby(['age', 'sn', 'inc_str', 'con_hit']).mean().reset_index()
    df = df[df['inc_str'] != 'neutral']
    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    print(df)
    print(list(df.columns))
    print(df['obj'])
    quit()

    from pymer4.models import Lmer
    for ROI in ROI_cols:

        print(f'ROI: {ROI}')
        formula = f'{ROI} ~ age*inc_str*con_hit + (1|sn)'
        # print(f'Formula: {formula}')

        keys = keys_from_formula(formula)
        # print(f'Keys: {keys}')

        model = Lmer(formula, data=df)
        try:
            model.fit(REML=True, verbose=False, summary=False)
        except Exception as e:
            print(f'Error fitting model: {e}')
            continue


        summary = model.coefs
        print(summary.iloc[1:])
        t = summary['T-stat'].loc['(Intercept)']
        p = summary['P-val'].loc['(Intercept)']
        # print(model.fit())
        # quit()


if __name__ == '__main__':
    # prep_activation_for_univariate('bl3_fMRI')
    do_univariate_analysis()

