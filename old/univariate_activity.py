import os

from tqdm import tqdm

os.chdir(r'C:\PycharmProjects\SchemeRep')

from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests

from Utils.atlas_funcs import get_atlas
from Study1A.load_Study1A_funcs import get_ROI_vecs
from old.plot_gen import my_plot_surf
from organize_bhv import get_trial_info
from org_sns import get_sns, get_shenyang_subjects
import numpy as np

from Utils.pickle_wrap_funcs import pickle_wrap
from collections.abc import Iterable as iterable
import pandas as pd
from pandas.api.types import is_numeric_dtype
import scipy.stats as stats
import warnings

# suppress FutureWarning with "DataFrame.applymap has been deprecated"
warnings.filterwarnings('ignore',
                        message='DataFrame.applymap has been deprecated. '
                                'Use DataFrame.map instead.')
warnings.filterwarnings('ignore',
                        message='Error while trying to convert the column '
                                '"obj". Fall back to string conversion.')

def prep_activation_for_univariate(fp='obj3_fMRI', key='inc', conds=(1, 3),
                                   keys=('inc', 'con_hit'),
                                   only_sh_sns=False,
                                   combine_regions=False
                                   ):
    atlas = get_atlas(combine_regions=False, combine_bilateral=False,
                      shenyang=True)
    coords = atlas['coords']
    sh_sns = get_shenyang_subjects()
    age2sn = get_sns(fp, sh=only_sh_sns)

    ROI2age2inc2l = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    df_l = []
    ROI_l = None
    ROI_cols = None
    for i, age in enumerate([1, 2]):
        sns = age2sn[age]
        for sn in sns:
            print(f'Prepping univariate: {sn=}')
            df_sn = get_trial_info(sn, easy_override=False)
            ROI2vecs0 = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=False,
                                     easy_override=True,
                                     combine_regions=combine_regions)
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
    fp = r'H:\PycharmProjects_H\SchemeRep\Shenyang_R\shenyang_conceptual_memory.csv'
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
    df, ROI_cols = pickle_wrap(prep_activation_for_univariate, None, kwargs=kwargs, easy_override=False,
                               cache_dir='../cache')
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
    df, ROI_cols = pickle_wrap(prep_activation_for_univariate, None, kwargs=kwargs, easy_override=False,
                               cache_dir='../cache')
    df['cat'] = 'obj'
    kwargs = {'fp': 'scn7_fMRI', 'key': 'inc', 'conds': (1, 3),
              'only_sh_sns': True}
    # df = df[df['sn'] == '102']
    # print(df['pSTS_L_2_1'])
    # quit()

    df_scn, ROI_cols = pickle_wrap(prep_activation_for_univariate, None, kwargs=kwargs, easy_override=False,
                                   cache_dir='../cache')
    df_scn['cat'] = 'scn'
    df = pd.concat([df, df_scn], axis=0)

    df = df[~pd.isna(df['con_hit'])]
    df = df[~pd.isna(df['per_inc'])]
    df['con_hit'] = df['con_hit'].apply(lambda x: 'Hit' if x else 'Miss')
    df = include_shenyang_memory(df)
    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    df['obj'] = df['obj'].astype(str)
    df.reset_index(inplace=True)

    df_all = df
    for age in ['YA', 'OA', ]:
        ts = []
        ts_obj = []
        ts_scn = []
        df = df_all[df_all['age'] == age]
        for ROI in ROI_cols:

            obj_s = df[df['cat'] == 'obj'].groupby('sn')[ROI].mean()
            scn_s = df[df['cat'] == 'scn'].groupby('sn')[ROI].mean()
            M_obj = obj_s.mean()
            M_scn = scn_s.mean()
            t_obj, _ = stats.ttest_1samp(obj_s, 0)
            t_scn, _ = stats.ttest_1samp(scn_s, 0)
            t, _ = stats.ttest_rel(obj_s, scn_s)
            dif = obj_s - scn_s
            test = pd.concat([obj_s, scn_s, dif], axis=1)
            # print(test)
            print(f'{ROI} | {t_obj=:.2f}, {t_scn=:.2f}, {t=:.2f}, '
                  f'[{M_obj=:.3f}, {M_scn=:.3f}]')
            # quit()
            ts_obj.append(t_obj)
            ts_scn.append(t_scn)
            ts.append(t)

        # obj x scn correlation (244 ROIs) is r = .93
        df_cross = pd.DataFrame({'obj': ts_obj, 'scn': ts_scn})
        df_cross.dropna(inplace=True)
        print(f'{len(df_cross)=}')
        cross, p = stats.spearmanr(df_cross['obj'], df_cross['scn'])
        print(f'{age=}, {cross=:.2f}, {p=:.3f}')
        # quit()

        age2str = {'OA': 'Older adults', 'YA': 'Younger adults'}
        atlas = get_atlas(combine_regions=False)
        fp_pic = f'mass_ttest/scn_vs_obj_{age}.png'
        vmax = np.nanquantile(np.abs(ts), 0.95)
        # thresh = np.nanquantile(np.abs(ts), 0.5)
        thresh = 2
        ts = np.array(ts)
        my_plot_surf(ts, atlas, f'Object vs. Scene ({age2str[age]}), t-test',
                     fp_out=fp_pic, neg='Scn', pos='Obj',
                     vmax=vmax, thresh=thresh)

        fp_pic = f'mass_ttest/obj_beta_{age}.png'
        vmax = np.nanquantile(np.abs(ts_obj), 0.95)
        ts_obj = np.array(ts_obj)
        my_plot_surf(ts_obj, atlas, f'Object betas ({age2str[age]}), t-test',
                        fp_out=fp_pic, neg='Neg', pos='Pos',
                        vmax=vmax, thresh=thresh)

        fp_pic = f'mass_ttest/scn_beta_{age}.png'
        vmax = np.nanquantile(np.abs(ts_scn), 0.95)
        ts_scn = np.array(ts_scn)
        my_plot_surf(ts_scn, atlas, f'Scene betas ({age2str[age]}), t-test',
                        fp_out=fp_pic, neg='Neg', pos='Pos',
                        vmax=vmax, thresh=thresh)


def simple_effect(df, ROI_cols, key, fp):
    df_grp = df.groupby(['age', 'sn', 'inc'])[ROI_cols].mean()
    ages = ['healthy']
    age2ts = {'OA': [], 'YA': [], 'healthy': []}
    age2ps = {'OA': [], 'YA': [], 'healthy': []}

    for ROI in ROI_cols:
        for age in ages:
            if age == 'healthy':
                age_keys = ['YA', 'OA']
            else:
                age_keys = [age]
            # else:
                # age_keys =
            # print(df_grp)
            # quit()
            t, p = stats.ttest_rel(df_grp.loc[age_keys, :, 1][ROI],
                                   df_grp.loc[age_keys, :, 3][ROI],
                                   nan_policy='omit')
            age2ts[age].append(t)
            age2ps[age].append(p)
            if p < .01:
                print(f'{ROI=}, {age=}, {t=:.2f}, {p=:.3f}')
    atlas = get_atlas(combine_regions=False)
    for age in ages:
        ts = np.array(age2ts[age])
        sigs, p_corr, alpha_sidak, alpha_bon = \
            multipletests(age2ps[age], alpha=.05, method='fdr_bh')
        vmax = 4#np.min(np.abs(ts[sigs]))
        print(f'FDR corrected, {age}: {ts[sigs]=}')
        fp_pic = f'mass_ttest/activity/{fp}_{key}_{age}.png'
        Path(fp_pic).parent.mkdir(parents=True, exist_ok=True)
        fp2title = {'obj7_fMRI': 'Object betas',
                    'scn7_fMRI': 'Scene betas',
                    'con7_fMRI': 'Conceptual retrieval betas',
                    'vis7_fMRI': 'Visual retrieval betas'}
        my_plot_surf(ts, atlas,
                     f'{fp2title[fp]}\n{age}, Congruent vs. Incongruent',
                     fp_out=fp_pic, neg='Con', pos='Inc',
                     thresh=2.85, vmax=vmax)

def plot_congruency_lmer(fp='obj7_fMRI', do_lm=True, use_neu=True, fwe=False):
    pd.set_option('display.max_rows', 115)
    kwargs = {'fp': fp, 'key': 'inc', 'conds': (1, 2, 3),
              'only_sh_sns': False, 'combine_regions': False}
    df, ROI_cols = pickle_wrap(prep_activation_for_univariate, None,
                               kwargs=kwargs, easy_override=False,
                               cache_dir='cache')
    df.reset_index(inplace=True)
    df['obj'] = df['obj'].astype(str)
    ts = []
    ps = []

    os.environ['R_HOME'] = r'C:\Users\Paul\anaconda3\envs\py312\Lib\R'
    import statsmodels.formula.api as smf

    pd.set_option('display.precision', 5)

    if not use_neu:
        df = df[df['inc'] != 2]


    # inc_mapper = {1: 'inc', 2: 'neu', 3: 'con'}
    # df['inc'] = df['inc'].map(inc_mapper)
    df_grp = df.groupby(['sn', 'inc'])[ROI_cols].mean()
    # print(df_grp)
    # quit()
    # df_grp_ = df.groupby(['inc', 'sn'])[ROI_cols].mean()

    # print(df_grp.xs('inc', level='inc'))
    # print(type(df_grp))
    # print(df_grp.index[0])
    # print(df_grp_.loc[slice(None, '102')])
    # quit()

    # for ROI in ROI_cols:
    #     inc_vals = df_grp.xs(1, level='inc')[ROI]
    #     n = len(inc_vals)
    #     print(f'{n=}')
    #     con_vals = df_grp.xs(3, level='inc')[ROI]
    #     dif = con_vals - inc_vals
    #     M = dif.mean()
    #     t, p = stats.ttest_1samp(dif, 0)
    #     print(f'{ROI} | {t=:.2f}, {p=:.5f}, {M=:.3f}')
    # quit()

    df_grp = df_grp.reset_index()
    df_grp['inc_str'] = df_grp['inc'].apply(lambda x: 'inc' if x == 1 else
                                                 'neu' if x == 2 else 'con')

    for ROI in tqdm(ROI_cols):
        if do_lm:
            formula = f'{ROI} ~ inc + sn'
            keys = keys_from_formula(formula)
            model = smf.ols(formula=formula, data=df_grp[keys].dropna())
            res = model.fit()
            p = res.pvalues.loc['inc']
            t = res.tvalues.loc['inc']
            ts.append(-t)
            ps.append(p)
            if p < .005:
                corr = .05 / p
                super_signif = corr > len(ROI_cols)
                # print(f'{ROI=} | {t=:.2f}, {p=:.5f}, {corr=:.1f}, '
                #       f'{super_signif=}')

            if use_neu:
                formula = f'{ROI} ~ sn + inc_str'
                keys = keys_from_formula(formula)
                model = smf.ols(formula=formula, data=df_grp[keys].dropna())
                res = model.fit()
                p_neu = res.pvalues.loc['inc_str[T.neu]']
                neu_ef = res.params['inc_str[T.neu]']
                inc_ef = res.params['inc_str[T.inc]']
                if p > .005 and p_neu < .005:
                    print(f'Neu ROI: {ROI=} | {neu_ef=:.3f} '
                          f'({p_neu=:.4f}), {inc_ef=:.3f}')
        else:
            from pymer4.models import Lmer
            formula = f'{ROI} ~ inc + (1 + inc | sn) + (1 | obj)'
            keys = keys_from_formula(formula)
            model = Lmer(formula, data=df[keys].dropna())
            model.fit(verbose=False, summarize=False)
            res = model.coefs
            t, p = res.loc['inc', ['T-stat', 'P-val']]
            ts.append(-t)
            ps.append(p)
            if p < .005:
                res['corr'] = .05 / res['P-val']
                res['FWE'] = res['corr'] > len(ROI_cols)
                res['FWE'] = res['FWE'].apply(lambda x: 'Super signif' if x else '')
                print(f'{ROI=}')
                print(res[['T-stat', 'P-val', 'corr']].loc['inc'])
                print('-'*50)

    ts = np.array(ts)

    sigs, p_corr, alpha_sidak, alpha_bon = (
        multipletests(ps, alpha=.05, method='holm-sidak' if fwe else 'fdr_bh'))

    # print(ts)
    # print('-----------')
    # print(f'{sigs=}')
    # print(ts[sigs])
    # print(f'FDR corrected: {ts[sigs]=}')


    atlas = get_atlas()
    er_str = '' if do_lm else 'er'
    neu_str = '' if use_neu else '_NoNeu'
    fwe_str = '_FWE' if fwe else ''
    fp_pic = (rf'result_pics/activity/{fp}_inc_healthy_lm{er_str}{neu_str}'
              rf'{fwe_str}.png')
    vmin = np.min(np.abs(ts[sigs])) - .01
    if vmin > 4:
        vmax = vmin + 1
    else:
        vmax = 4
    # print(f'{vmin=}')
    # vmin = min(vmin, 3)
    # print(f'{ts=}')
    # print(len(ts))
    # vmin = 2
    title = f'Activation ~ congruency (FDR)' if use_neu \
        else f'Activation: congruency vs. incongruency (FDR)'
    my_plot_surf(ts, atlas,
                 title,
                 fp_out=fp_pic, neg='Con', pos='Inc',
                 thresh=vmin, vmax=vmax)
    # quit()
    regions = atlas['ROI_regions_laterality']
    ROIs = atlas['ROIs']
    region_in = set()
    for i, region in enumerate(regions):
        if sigs[i]:
            print(f'Sig: {ROIs[i]} | {region}: {ps[i]=}')
            pn = 'p' if ts[i] > 0 else 'n'
            region_in.add(f'{region}_{pn}')
    print(f'{region_in=}')
    ts_clean = []
    for i, region in enumerate(regions):
        pn = 'p' if ts[i] > 0 else 'n'
        if f'{region}_{pn}' in region_in:
            ts_clean.append(ts[i])
        else:
            ts_clean.append(0)

    fp_pic = (f'result_pics/activity/{fp}_inc_healthy_lm{er_str}{neu_str}'
              f'{fwe_str}_low_thresh.png')
    title = title.replace('FDR', 'FDR+')
    my_plot_surf(ts_clean, atlas,
                 title,
                 fp_out=fp_pic, neg='Con', pos='Inc',
                 thresh=2.0, vmax=4)

def do_univariate_analysis(fp='cmb3_fMRI'):
    pd.set_option('display.max_rows', 115)
    kwargs = {'fp': fp, 'key': 'inc', 'conds': (1, 3),
              'only_sh_sns': True, 'combine_regions': False}
    df, ROI_cols = pickle_wrap(prep_activation_for_univariate, None,
                               kwargs=kwargs, easy_override=False,
                               cache_dir='cache')

    # df = df[~pd.isna(df['con_hit'])]
    # df = df[~pd.isna(df['per_inc'])]
    # df['con_hit'] = df['con_hit'].apply(lambda x: 'Hit' if x else 'Miss')
    # df = include_shenyang_memory(df)
    df['age'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    df['obj'] = df['obj'].astype(str)
    df.reset_index(inplace=True)
    simple_effect(df, ROI_cols, kwargs['key'], fp)
    return
    from pymer4.models import Lmer
    for ROI in ROI_cols:
        # if 'ITG' not in ROI:
        #     continue
        # formula = f'{ROI} ~ age*per_inc14_str*con_hit + (1|sn) + (1|obj)'
        formula = f'{ROI} ~ age*inc + (1|sn) + (1|obj)'
        # formula = f'{ROI} ~ age*per_inc14_str*con_hit + (1|sn) + (1|obj)'

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
            continue
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
                       'obj3_fMRI': 'Object model betas',
                       'con7_fMRI': 'Conceptual retrieval betas',
                       'vis7_fMRI': 'Visual retrieval betas'}
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
    do_univariate_analysis(fp='obj7_fMRI')
    # plot_congruency_lmer(fp='obj7_fMRI')


