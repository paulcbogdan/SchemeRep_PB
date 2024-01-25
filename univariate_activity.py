from collections import defaultdict

from matplotlib import pyplot as plt

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from old.plot_gen import my_plot_surf
from organize_bhv import get_trial_info
from org_sns import get_all_sns, get_shenyang_subjects
import numpy as np

from utils import pickle_wrap
from collections.abc import Iterable as iterable
import pandas as pd
from pandas.api.types import is_numeric_dtype

def prep_activation_for_univariate(fp='obj3_fMRI', key='inc', conds=(1, 3),
                                   keys=('inc', 'con_hit'),
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
        for sn in sns:
            print(f'Prepping univariate: {sn=}')
            df_sn = get_trial_info(sn, easy_override=True)
            ROI2vecs0 = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=False,
                                     easy_override=True,
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
    df['con_hit'] = df.apply(lambda x: d[(x['sn'], x['obj'])] if
                     (x['sn'], x['obj']) in d else np.nan, axis=1)
    return df

def do_univariate_living(fp='bl3_fMRI'):
    pd.set_option('display.max_rows', 115)
    # kwargs = {'fp': fp,
    #           'split': False,
    #           'key': 'living',
    #           'conds': (False, True),
    #           }
    kwargs = {'fp': fp, 'key': 'inc', 'conds': (1, 3),
              'only_sh_sns': True}
    df, ROI_cols = pickle_wrap(None, prep_activation_for_univariate,
                                kwargs=kwargs, cache_dir='cache',
                                easy_override=False)
    # df = include_shenyang_memory(df)
    # df['con_hit'] = df['con_hit'].apply(lambda x: 'Hit' if x else 'Miss')
    # print(df['con_hit'].value_counts())
    # for sn, df_sn in df.groupby('sn'):
    #     print(f'{sn=}')
    #     print(df_sn['con_hit'].value_counts())
    #     print('--------------------------')
    # quit()
    # df = include_shenyang_memory(df)
    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    df['obj'] = df['obj'].astype(str)
    df.reset_index(inplace=True)

    from pymer4.models import Lmer
    for ROI in ROI_cols:
        print('-'*100)
        print(f'{ROI=}')
        # if 'ITG' not in ROI:
        #     continue
        # if 'EVC_R_5_1' not in ROI:
        #     continue
        formula = f'{ROI} ~ living + (1|sn)'
        keys = keys_from_formula(formula)
        model = Lmer(formula, data=df[keys].dropna())
        try:
            model.fit(REML=True, verbose=False, summary=False)
        except Exception as e:
            print(f'Error fitting model: {e}')
            continue
        print(model.summary())
        result = model.anova()
        print(result)

def do_obj_vs_scn():
    pd.set_option('display.max_rows', 115)
    kwargs = {'fp': 'obj7_fMRI', 'key': 'inc', 'conds': (1, 3),
              'only_sh_sns': True}
    df, ROI_cols = pickle_wrap(None, prep_activation_for_univariate,
                               kwargs=kwargs, cache_dir='cache',
                               easy_override=False)
    df['cat'] = 'obj'
    kwargs = {'fp': 'scn7_fMRI', 'key': 'inc', 'conds': (1, 3),
              'only_sh_sns': True}
    df_scn, ROI_cols = pickle_wrap(None, prep_activation_for_univariate,
                                   kwargs=kwargs, cache_dir='cache',
                                   easy_override=False)
    df_scn['cat'] = 'scn'
    df = pd.concat([df, df_scn], axis=0)

    df = df[~pd.isna(df['con_hit'])]
    df = df[~pd.isna(df['per_inc'])]
    df['con_hit'] = df['con_hit'].apply(lambda x: 'Hit' if x else 'Miss')
    df = include_shenyang_memory(df)
    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    df['obj'] = df['obj'].astype(str)
    df.reset_index(inplace=True)
    from pymer4.models import Lmer
    ts = []
    for ROI in ROI_cols:
        # if 'PhG' not in ROI and 'FuG' not in ROI:
        #     continue

        # if 'EVC_R_5_1' not in ROI:
        #     continue
        formula = f'{ROI} ~ cat + (1 + cat |sn)'

        # formula = f'{ROI} ~ age*per_inc14_str*con_hit + (1|sn) + (1|obj)'
        keys = keys_from_formula(formula)
        model = Lmer(formula, data=df[keys].dropna())
        try:
            model.fit(REML=True, verbose=False, summary=False)
        except Exception as e:
            print(f'Error fitting model: {e}')
            continue
        res = model.coefs
        ts.append(res['T-stat'].iloc[1])
        if res['P-val'].iloc[1] < .01:
            print(f'Sig: {ROI=}')
            print(model.summary())
        else:
            print('Insignificant')
    atlas = get_atlas(combine_regions=False)
    fp_pic = f'mass_ttest/scn_vs_obj.png'
    vmax = np.nanquantile(np.abs(ts), 0.95)
    # thresh = np.nanquantile(np.abs(ts), 0.5)
    thresh = 3
    ts = np.array(ts)
    my_plot_surf(ts, atlas, 'Object vs. Scene',
                 fp_out=fp_pic, neg='Obj', pos='Scn',
                 vmax=vmax, thresh=thresh)

def do_univariate_analysis(fp='cmb3_fMRI'):
    pd.set_option('display.max_rows', 115)
    kwargs = {'fp': 'obj7_fMRI', 'key': 'inc', 'conds': (1, 3),
              'only_sh_sns': True}
    df, ROI_cols = pickle_wrap(None, prep_activation_for_univariate,
                                kwargs=kwargs, cache_dir='cache',
                                easy_override=False)

    df = df[~pd.isna(df['con_hit'])]
    df = df[~pd.isna(df['per_inc'])]
    df['con_hit'] = df['con_hit'].apply(lambda x: 'Hit' if x else 'Miss')
    df = include_shenyang_memory(df)
    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    df['obj'] = df['obj'].astype(str)
    df.reset_index(inplace=True)
    from pymer4.models import Lmer
    for ROI in ROI_cols:
        if 'ITG' not in ROI:
            continue

        formula = f'{ROI} ~ age*per_inc14_str*con_hit + (1|sn) + (1|obj)'
        keys = keys_from_formula(formula)
        model = Lmer(formula, data=df[keys].dropna())
        try:
            model.fit(REML=True, verbose=False, summary=False)
        except Exception as e:
            print(f'Error fitting model: {e}')
            continue
        result = model.anova()
        if True:
            plt.rcParams.update({'font.size': 12})
            effect2name = {'con_hit': 'Main effect of conceptual memory',
                           'age': 'Main effect of age',
                           'per_inc14_str': 'Main effect of congruency',
                           'age:per_inc14_str': 'Interaction age x congruency',
                           'age:con_hit': 'Interaction age x conceptual memory',
                           'per_inc14_str:con_hit': 'Interaction congruency x conceptual memory',
                           'age:per_inc14_str:con_hit': '3-way interaction'}

            print('-'*100)
            print(f'ROI: {ROI}')
            print(result)
            effect = result['P-val'].idxmin()
            effect = effect2name[effect]
            # quit()
            df_grp = df.groupby(['sn', 'age', 'per_inc14_str', 'con_hit'])[ROI].mean()
            df_grp_err = df_grp.groupby(['age', 'per_inc14_str', 'con_hit']).sem() * 1.96
            df_grp_err = df_grp_err.sort_index(axis=0, level=(0, 1),
                                      ascending=False)
            df_grp_M = df_grp.groupby(['age', 'per_inc14_str', 'con_hit']).mean()
            df_grp_M = df_grp_M.sort_index(axis=0, level=(0, 1),
                                        ascending=False)

            plt.bar([-0.15, .85, 1.85, 2.85], df_grp_M.loc[:, :, 'Hit'],
                    yerr=df_grp_err.loc[:, :, 'Hit'], width=0.25,
                    label='Hit', color='dodgerblue')
            plt.bar([0.15, 1.15, 2.15, 3.15], df_grp_M.loc[:, :, 'Miss'],
                    yerr=df_grp_err.loc[:, :, 'Miss'], width=0.25,
                    label='Miss', color='red')
            # plt.xlabel(['e', 'f', 'g', 'h'])
            plt.xticks([0, 1, 2, 3], ['(Inc1)\nYA', '(Con4)\nYA',
                                      '(Inc1)\nOA', ' (Con4)\nOA'])

            plt.legend(frameon=False, loc='upper center')
            plt.ylabel(f'Mean beta: {ROI}')
            fp2name = {'cmb3_fMRI': 'Scene + object combined model betas',
                          'scn3_fMRI': 'Scene model betas',
                          'obj3_fMRI': 'Object model betas'}
            fp_name = fp2name[fp]
            title_str = f'{fp_name}\nResult: {effect}'
            plt.title(title_str)
            plt.tight_layout()
            plt.show()
            # quit()
        else:
            print(f'ROI: {ROI} - No significance')
        #     print('Has significance!')
        return

if __name__ == '__main__':
    # prep_activation_for_univariate('bl3_fMRI')
    # do_univariate_living()
    do_obj_vs_scn()

    # do_univariate_analysis(fp='con3_fMRI')


