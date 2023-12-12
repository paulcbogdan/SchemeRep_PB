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
    formula = formula.split('~')[0]
    formula = formula.split('+')
    formula = [x.strip() for x in formula]
    return formula

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
    kwargs = {'fp': fp, 'key': 'inc', 'conds': (1, 3),
              'only_sh_sns': True}
    df, ROI_cols = pickle_wrap(None, prep_activation_for_univariate,
                                kwargs=kwargs, cache_dir='cache',
                                easy_override=False)
    # print(df['per_inc14_str'])
    # quit()
    # print(df['sn_sh'].unique())
    # quit()
    # df = df[~pd.isna(df['per_inc14_str'])]
    # df = df[df['con_hit'] > 0]
    df = df[~pd.isna(df['con_hit'])]
    df = df[~pd.isna(df['per_inc'])]
    # df = df[(df['per_inc'] == 1) | (df['per_inc'] == 4)]
    # df_test = df[df['sn'] == '104']
    # print(df_test['con_hit'])
    # sns_sh = get_shenyang_subjects()
    # print(f'{len(sns_sh)=}')
    # sns_mine = set(df['sn'].unique())
    # # for sn in sns_sh:
    # #     if sn not in sns_mine:
    # #         print(f'shenyang\'s {sn} not in mine')
    # print('-')
    # for sn in sns_mine:
    #     if sn not in sns_sh:
    #         print(f'my {sn} not in shenyang\'s')

    sns = df['sn'].unique()
    # print(f'{len(df) + 112 =}')
    df.dropna(subset=['ATL_R_6_5'], inplace=True)
    df.sort_values(by=['sn', 'obj'], inplace=True)
    df.reset_index(inplace=True)

    df['con_hit'] = df['con_hit'].apply(lambda x: 'Hit' if x else 'Miss')
    df = include_shenyang_memory(df)
    # print(df['con_hit'].value_counts())
    # quit()

    # df[['sn', 'obj', 'con_hit']].to_csv('C:\PycharmProjects_C\SchemeRep\Shenyang_R\pandas_df.csv')
    # quit()
    # df = df[df['sn'] == '105']
    # pd.set_option('display.max_rows', 115)
    # print(df[['obj', 'con_resp']])
    # # quit()
    #
    # # print(df['con_hit'].sum())
    #
    # for idx, row in df[['sn', 'obj', 'con_hit', 'con_resp', 'enc_run']].iterrows():
    #     print(dict(row))
    #     if idx > 100:
    #         break
    # quit()


    # print(len(sns))
    # print(f'{len(df)=}')
    # sns_str = ' '.join(sns)
    # print(sns_str)

    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    #
    from pymer4.models import Lmer
    #
    ROI = 'ATL_R_6_5'
    # formula = f'{ROI} ~ age + (1|sn) + (1|obj)'
    # formula = f'{ROI} ~ con_hit + (1|sn) + (1|obj)'
    # formula = f'{ROI} ~ per_inc14_str + (1|sn) + (1|obj)'
    df = df.dropna(subset=['per_inc14_str'])
    df.sort_values(by=['con_hit', 'per_inc14_str', 'age'], inplace=True,
                    ascending=True)
    formula = f'{ROI} ~ age*per_inc14_str*con_hit + (1|sn) + (1|obj)'

    # df['Subject'] = df['sn']
    # df['Object'] = df['obj']
    # df['AgeGrp'] = df['age']
    # df['RCON_Old'] = df['con_hit']
    # df['Cong_subj_1_4'] = df['per_inc14_str']
    #
    # formula = f'{ROI} ~ 1 + AgeGrp * RCON_Old * Cong_subj_1_4 + ' \
    #           f'(1|Subject) + (1|Object)'
    model = Lmer(formula, data=df)
    summary = model.fit(REML=True, verbose=True, summary=True,
                        control="optimizer='Nelder_Mead', "
                                "optCtrl = list(FtolAbs=1e-8, XtolRel=1e-8)")
    print(summary)
    print(model.anova())
    quit()


    # formula = f'{ROI} ~ con_hit + (1|sn) + (1|obj)'
    # # formula = f'{ROI} ~ age + (1|sn) + (1|obj)'
    # model = Lmer(formula, data=df)
    # summary = model.fit(REML=True, verbose=True, summary=True)
    #
    # df[['ATL_R_6_5', 'age', 'per_inc14_str', 'con_hit', 'sn', 'obj']].to_csv(
    #     'C:\PycharmProjects_C\SchemeRep\Shenyang_R\pandas_df.csv')

    # quit()

    # print(df)
    # print(list(df.columns))
    # print(df['obj'])
    # quit()
    # print(len(df))
    # quit()

    # 102 103 105 106 107 108 110 111 112 114 116 117 120 123 124 125 126 127 128 130 132 134 135 136 137
    # 102 103 104 105 106 107 108 109 110 111 112 113 114 115 117 118 119 120 123 124 126 127 128 129 130 131 132 134 136 137 201 202 203 204 205 206 207 208 209 210 211 214 216 217 218 219 221 222 225 227 230 232 233 235

    for ROI in ROI_cols:
        print(f'ROI: {ROI}')
        # if ROI != 'ATL_R_6_5':
        #     continue
        formula = f'{ROI} ~ age*per_inc14_str*con_hit + (1|sn) + (1|obj)'


        keys = keys_from_formula(formula)
        # print(f'Keys: {keys}')

        model = Lmer(formula, data=df)
        try:
            model.fit(REML=True, verbose=True, summary=True)
        except Exception as e:
            print(f'Error fitting model: {e}')
            continue
        print(model.anova())
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

