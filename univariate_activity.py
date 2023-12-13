from collections import defaultdict

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_trial_info
from org_sns import get_all_sns, get_shenyang_subjects
import numpy as np

from utils import pickle_wrap
from collections.abc import Iterable as iterable
import pandas as pd
from pandas.api.types import is_numeric_dtype

def prep_activation_for_univariate(fp='obj3_fMRI', key='inc', conds=(1, 3),
                                   keys=('inc', 'con_hit'),
                                   # conds=((1, 3), (False, True))
                                   only_sh_sns=False
                                   ):
    atlas = get_atlas(combine_regions=False, combine_bilateral=False,
                      shenyang=True)
    coords = atlas['coords']
    sh_sns = get_shenyang_subjects()
    age2sn = get_all_sns(fp, sh=only_sh_sns)

    ROI2age2inc2l = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    df_l = []
    ROI_l = None
    ROI_cols = None
    for i, age in enumerate([1, 2]):
        # if age != 1:
        #     continue
        sns = age2sn[age]
        # sns = ['102']
        for sn in sns:
            print(f'Prepping univariate: {sn=}')
            df_sn = get_trial_info(sn, easy_override=True)
            # print(df_sn['obj'])
            # print(fp)
            # print(df_sn[fp].iloc[7])
            # quit()
            ROI2vecs0 = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=False,
                                     easy_override=True,
                                     combine_regions=False)
            if ROI_l is None:
                ROI_l = list(ROI2vecs0.keys())
            ar = []

            # ROI_l = ['1 SFG_L_7_1']
            for ROI in ROI_l:
                activity = ROI2vecs0[ROI]
                ar.append(np.nanmean(activity, axis=1))
                # print(activity.shape)
                # print(activity[7])
                # print(ar[-1])
                # quit()
                ROI_cols = ['_'.join(x.split()[1:]) for x in ROI_l]
            # print(ar)
            df_sn_ROI = pd.DataFrame(np.array(ar).T, columns=ROI_cols)
            df_sn = pd.concat([df_sn, df_sn_ROI], axis=1)
            df_sn['age'] = age
            df_sn['sn_sh'] = sn in sh_sns
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
    DV, IV = formula.split('~')
    DV = DV.replace(' ', '')
    for key in ['*', '+', '(', '|', ')']:
        IV = IV.replace(key, ' ')
    while True:
        IV = IV.replace('  ', ' ')
        if '  ' not in IV:
            break
    IV = IV.split()
    IV = [x for x in IV if (x and x != '1')]
    return [DV] + IV

def mean_or_fist(l):
    if is_numeric_dtype(l):
        return l.mean()
    else:
        return l.iloc[0]

def include_shenyang_memory(df):
    fp = r'C:\PycharmProjects_C\SchemeRep\Shenyang_R\shenyang_conceptual_memory.csv'
    df_sh = pd.read_csv(fp)
    d = defaultdict(None)
    for idx, row in df_sh.iterrows():
        d[(str(row['Subject']), row['Object'])] = row['RCON_Old']

    d_test = list((x, y) for x, y in df[['sn', 'obj']].values)
    # print(len(d_test))
    # quit()
    df['con_hit'] = df.apply(lambda x: d[(x['sn'], x['obj'])], axis=1)
    return df

def do_univariate_analysis(fp='obj3_fMRI'):
    pd.set_option('display.max_rows', 115)
    kwargs = {'fp': fp, 'key': 'inc', 'conds': (1, 3),
              'only_sh_sns': True}
    df, ROI_cols = pickle_wrap(None, prep_activation_for_univariate,
                                kwargs=kwargs, cache_dir='cache',
                                easy_override=True)
    df = df[~pd.isna(df['con_hit'])]
    df = df[~pd.isna(df['per_inc'])]
    df['con_hit'] = df['con_hit'].apply(lambda x: 'Hit' if x else 'Miss')
    df = include_shenyang_memory(df)
    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')

    # df = df[df['sn'] == '102']
    # df_pruned = df[['sn', 'obj', 'SFG_L_7_1']]
    # print(df_pruned)
    # quit()

    from pymer4.models import Lmer
    for ROI in ROI_cols:
        print(f'ROI: {ROI}')
        # if ROI != 'ATL_R_6_5':
        #     continue
        formula = f'{ROI} ~ age*per_inc14_str*con_hit + (1|sn) + (1|obj)'
        keys = keys_from_formula(formula)
        model = Lmer(formula, data=df[keys].dropna())
        try:
            model.fit(REML=True, verbose=True, summary=True)
        except Exception as e:
            print(f'Error fitting model: {e}')
            continue
        result = model.anova()
        print(result)
        # print(result['Sig'].values)
        # if '***' in result['Sig'].values:
        #     print('Has significance!')


if __name__ == '__main__':
    # prep_activation_for_univariate('bl3_fMRI')
    do_univariate_analysis()

