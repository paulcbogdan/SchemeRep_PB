from collections import defaultdict
from pathlib import Path

import pandas as pd
from pymer4 import Lmer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GroupKFold
from sklearn.svm import SVC
from tqdm import tqdm

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from network_funcs import load_FC_for_Lifu
from classifiers import stratify
from old.plot_gen import my_plot_surf
from org_sns import get_all_sns
from organize_bhv import get_trial_info
from univariate_activity import keys_from_formula
from utils import pickle_wrap, stdize, regress_out
import numpy as np

from functools import wraps
from time import time
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings

# pandas suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def timing(f):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

@timing
def do_activity_ERS_sn(sn, fp0='bl3_fMRI', fp1='obj7_fMRI',
                       combine_regions=False):
    atlas = get_atlas(combine_regions=combine_regions)
    df_sn = get_trial_info(sn)
    kwargs0 = {
        'sn': sn,
        'atlas': atlas,
        'fp_fMRI_col': fp0,
        'df_sn': df_sn,
        'nan_thresh': 1.01,
        'drop_nan_voxels': False,
        'org_by_region': False,
        'easy_override': False,
        'combine_regions': combine_regions,
    }
    ROI2vecs0 = get_ROI_vecs(**kwargs0)
    kwargs1 = kwargs0 | {'fp_fMRI_col': fp1}
    ROI2vecs1 = get_ROI_vecs(**kwargs1)
    if combine_regions:
        assert 'SFG_L' in ROI2vecs0, f'{ROI2vecs0.keys()=}'
        assert 'SFG_L' in ROI2vecs1, f'{ROI2vecs1.keys()=}'
    ERSs = []
    cols = []
    for ROI, vecs in ROI2vecs0.items():
        ROI2vecs0[ROI] = stdize(vecs, axis=0, nans=True)
        ROI2vecs1[ROI] = stdize(ROI2vecs1[ROI], axis=0, nans=True)

    for ROI, vecs0 in ROI2vecs0.items():
        vecs1 = ROI2vecs1[ROI]
        corr = vecs0[None, :, :] * vecs1[:, None, :]
        corr = np.nanmean(corr, axis=-1)
        same = np.diag(corr.copy()) # copy needed to avoid next line from NaNing
        corr[np.diag_indices_from(corr)] = np.nan
        dif = np.nanmean(corr, axis=0)
        ERS = same - dif
        ERSs.append(ERS)
        ROI_clean = ROI.replace(' ', '_')
        cols.append(f'ERS_{fp0[:2]}_{fp1[:2]}_{ROI_clean}')
    # setting df_sn[cols]=ERS.T instead gives a fragmentation warning
    df_cols = pd.DataFrame(np.array(ERSs).T, columns=cols, index=df_sn.index)
    df_sn = pd.concat([df_sn, df_cols], axis=1)
    df_sn['sn'] = sn
    return df_sn, cols

def get_df_ERS(fp0='scn7_fMRI', fp1='vis7_fMRI', combine_regions=True):
    age2sn = get_all_sns('all', sh=False)
    sns = age2sn['healthy']
    dfs = []
    ERS_cols = []
    ERS_cols_bilateral = []
    for sn in tqdm(sns, desc='Making ERS dfs'):
        cmb_str = '_cmb' if combine_regions else ''
        fp_pkl = f'cache/ERS_df_{sn}_{fp0}_{fp1}{cmb_str}_fixed.pkl'
        df_sn, ERS_cols = pickle_wrap(fp_pkl, lambda: do_activity_ERS_sn(sn,
                combine_regions=combine_regions, fp0=fp0, fp1=fp1,),
                                      easy_override=False)
        ERS_cols_bilateral = sorted(list(set([col[:-2] for col in ERS_cols])))
        for col in ERS_cols_bilateral:
            try:
                df_sn[col] = (df_sn[f'{col}_L'] + df_sn[f'{col}_R']) / 2
            except KeyError:
                pass

        df_sn['sn'] = sn
        df_sn['age'] = int(sn[0])
        dfs.append(df_sn)
        # if len(dfs) > 5:
        #     break
    df = pd.concat(dfs, axis=0)
    return df, ERS_cols, ERS_cols_bilateral



def analyze_ERS(fp0='scn7_fMRI', fp1='bl7_fMRI', combine_regions=True):
    df, _, ERS_cols = get_df_ERS(fp0, fp1, combine_regions)
    df.set_index(['sn', 'obj'], inplace=True)
    df_ctrl, _, ERS_ctrl_cols = get_df_ERS('obj7_fMRI', fp1, combine_regions)
    df_ctrl.set_index(['sn', 'obj'], inplace=True)
    df = df.join(df_ctrl[ERS_ctrl_cols])
    df.reset_index(inplace=True)
    for col, ctrl_col in zip(ERS_cols, ERS_ctrl_cols):
        # df[col] = regress_out(df[ctrl_col], df[col])

    #     df_grp = df.groupby(['sn', 'inc'])[[col, ctrl_col]].mean()
    #     df_grp[col] = regress_out(df_grp[ctrl_col], df_grp[col])
    #     t0, p0 = stats.ttest_1samp(df_grp, 0, nan_policy='omit')
    #     print(f'{col} | Above zero (ctrl): {t0=:.2f}, {p0=:.3f} | '
    #           f'{t_age=:.2f}, {t_inc=:.2f}')
    #
    # # print(df)
    # # quit()
    #
    # for col in ERS_cols:
        df_grp = df.groupby(['sn', 'inc'])[col].mean()
        # df_grp = df.groupby(['sn', 'inc'])[col].mean()
        t_inc, p = stats.ttest_rel(df_grp.loc[:, 1], df_grp.loc[:, 3],
                               nan_policy='omit')
        df_age = df.groupby(['sn', 'age'])[col].mean()
        # df_grp = df.groupby(['sn', 'inc'])[col].mean()
        try:
            t_age, p = stats.ttest_ind(df_age.loc[:, 1], df_age.loc[:, 2],
                                   nan_policy='omit')
        except KeyError:
            t_age = 0
            pass
        df_grp = df.groupby(['sn'])[col].mean()
        t0, p0 = stats.ttest_1samp(df_grp, 0, nan_policy='omit')
        print(f'{col} | Above zero: {t0=:.2f}, {p0=:.3f} | '
              f'{t_age=:.2f}, {t_inc=:.2f}')
        # continue

        # if (p < .01) | (p0 < .01):
        #     # print(f'{col=}')
        #     # print(f'\t Inc effect: {t=:.2f}, {p=:.3f}')
        #     print(f'{col}, Above zero: {t0=:.2f}, {p0=:.3f}')
        #     # print(df_grp)
        # continue
        # formula = f'{col} ~ 1 + inc + (1 | sn)'
        # keys = keys_from_formula(formula)
        # model = Lmer(formula, data=df[keys].dropna())
        # model.fit(REML=True, verbose=False, summary=False)
        # result = model.summary()
        #
        # if (result['P-val'].iloc[0] < .01) | (result['P-val'].iloc[1] < .01):
        #     print(f'{col=}')
        #     print(result)
        #     print('-'*100)

@timing
def do_activity_MVPA_sn(sn, fp, key='inc', vals=(1, 3), combine_regions=False,
                        linear=True, nan_thresh=.75):
    atlas = get_atlas(combine_regions=combine_regions)
    df_sn = get_trial_info(sn)
    kwargs0 = {
        'sn': sn,
        'atlas': atlas,
        'fp_fMRI_col': fp,
        'df_sn': df_sn,
        'nan_thresh': 1.01,
        'drop_nan_voxels': False,
        'org_by_region': False,
        'easy_override': False,
        'combine_regions': combine_regions,
    }
    run_key = 'obj_run'
    ROI2vecs = get_ROI_vecs(**kwargs0)
    accs = []

    for ROI, vecs in ROI2vecs.items():
        X = []
        Y = []
        groups = []
        for i, val in enumerate(vals):
            groups.extend(list(df_sn.loc[df_sn[key] == val, run_key]))
            x_val = vecs[df_sn[key] == val, :]
            X.append(x_val)
            Y.extend([i] * x_val.shape[0])
        X = np.concatenate(X, axis=0)
        nan_voxels = np.isnan(X).any(axis=0)
        p_nan_voxel = np.mean(nan_voxels)
        if p_nan_voxel > nan_thresh:
            print(f'{ROI=} has {p_nan_voxel=} (> {nan_thresh:.2f})')
            accs.append(np.nan)
            continue
        X = X[:, ~nan_voxels]
        Y = np.array(Y)
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=0)
        clf = SVC(kernel='linear' if linear else 'rbf')
        acc = cross_val_score(clf, X, Y, cv=cv)
        acc = np.mean(acc)
        acc -= 0.5
        accs.append(acc)
    return accs, list(ROI2vecs.keys())

@timing
def do_activity_corr_sn(sn, fp, key='inc', vals=(1, 3), combine_regions=False,
                        linear=True, nan_thresh=.75):
    atlas = get_atlas(combine_regions=combine_regions)
    df_sn = get_trial_info(sn, easy_override=False)
    sess = fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').\
        replace('7', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)
    df_sn.reset_index(inplace=True)
    lowest_cnt = 100
    for run in range(3):
        # trial_low = run * len(df_sn) // 3
        # trial_high = (run + 1) * len(df_sn) // 3
        # df_run = df_sn.iloc[trial_low:trial_high, :]
        df_run = df_sn[df_sn[f'{sess}_run'] == run + 1]
        # df_run = df_run.sample(frac=1)
        cnt = df_run[key].value_counts()
        # print(cnt)
        lowest_cnt = min(lowest_cnt, cnt.min())
        df_sn.loc[df_run.index, 'cum_count'] = \
            df_run.groupby(key).cumcount()
    # print(df_sn['cum_count'])
    # quit()
    # print(f'{lowest_cnt=}')
    # lowest_cnt = 100
    # num_bads = sum(df_sn['cum_count'] >= lowest_cnt)
    pd.set_option('display.max_rows', None)

    # print(df_sn[[key, 'cum_count', f'{sess}_trial']])
    #
    # print(f'{num_bads=}')
    # if num_bads >
    df_sn.loc[df_sn['cum_count'] >= lowest_cnt, key] = np.nan # TODO: toggle
    # df_sn.iloc[:len(df_sn) // 3]
    # quit()

    # df_sn.loc[df_sn['inc_run_cnt'] > 9, 'inc_run'] = np.nan
    # print(df_sn[key].value_counts())
    # quit()

    kwargs0 = {
        'sn': sn,
        'atlas': atlas,
        'fp_fMRI_col': fp,
        'df_sn': df_sn,
        'nan_thresh': 1.01,
        'drop_nan_voxels': False,
        'org_by_region': False,
        'easy_override': False,
        'combine_regions': combine_regions,
    }
    ROI2vecs = get_ROI_vecs(**kwargs0)
    MVPA_cols = []
    MVPA_cols_simp = []
    MVPA_cols_alt = []
    for ROI, vecs in ROI2vecs.items():
        ROI_clean = ROI.replace(' ', '_')
        col = f'MVPA_{fp[:2]}_{ROI_clean}'
        col_simple = f'MVPA_{fp[:2]}_{ROI_clean}_simp'
        col_alt = f'MVPA_{fp[:2]}_{ROI_clean}_alt'
        MVPA_cols.append(col)
        MVPA_cols_simp.append(col_simple)
        MVPA_cols_alt.append(col_alt)
        val2variants = {}
        val2x_vals = {}
        val2M = {}
        # shuffler = np.arange(df_sn.shape[0])
        # np.random.shuffle(shuffler)
        # df_sn = df_sn.iloc[shuffler, :]
        bad_voxels = np.sum(np.isnan(vecs), axis=0)
        vecs = vecs[:, bad_voxels < vecs.shape[0] / 3 - 1]
        if vecs.shape[1] < 100:
            print(f'Few voxels | {sn}, {ROI}: {vecs.shape=}')

        for i, val in enumerate(vals):
            x_val = vecs[df_sn[key] == val, :]
            # print(f'{x_val.shape=}')
            # print(f'{x_val=}')
            # quit()


            # print(f'{x_val.shape=}')
            # n_vals = np.sum(~np.isnan(x_val), axis=0)
            # x_val = x_val[:, n_vals > 19]
            nans_per_trial = np.sum(np.isnan(x_val), axis=1)

            n_vals = np.sum(~np.isnan(x_val), axis=0)
            lowest_n_vals = np.min(n_vals)
            # if ROI == 'FuG_L':
            #     print(np.unique(n_vals))
            # print(ROI)
            # quit()
            # if lowest_n_vals < 10:
            #     print(f'Few trials | {sn}, {ROI}: {lowest_n_vals=} | '
            #           f'{np.mean(n_vals)=}')
            # if x_val.shape[1] < 100:
            #     print(f'Few voxels | {sn}, {ROI}: {x_val.shape=}')
            # if np.mean(n_vals) < 37:
            #     print(f'Some missing trials | {sn}, {ROI}: {np.mean(n_vals)=}')
            #     print(sorted(nans_per_trial))
            #     plt.imshow(np.isnan(x_val),
            #                extent=[0, x_val.shape[0], 0, x_val.shape[0]])
            #     plt.show()
            #     quit()
            # if x_val.shape[]
            # print(f'{x_val.shape=}')
            # print(x_val.shape)
            # quit()
            n_v_m = (n_vals - 1)# / n_vals
            M = np.nanmean(x_val, axis=0)
            M_variants = (M[None, :] * n_vals[None, :] - x_val * 1) / n_v_m[None, :]
            val2x_vals[val] = x_val
            # M_variants = x_val[0, :][None, :].copy() # TODO: stop
            # print(x_val.shape)
            # quit()
            val2variants[val] = M_variants
            val2M[val] = M
            np.set_printoptions(precision=3, suppress=True)

        ks = list(val2M.keys())
        assert val2M[ks[0]].shape == val2M[ks[1]].shape, \
            f'{val2M[1].shape=} {val2M[3].shape=}'

        for i, val in enumerate(vals):
            x_val = val2x_vals[val]
            # x_val[0, :] = np.nan
            M_variants = val2variants[val]
            # M_alt_l = []
            # for i_alt, val_alt in enumerate(vals):
            #     if val_alt == val:
            #         continue
            #     M_alt = val2M[val_alt]
            #     M_alt_l.append(M_alt)
            # M_alt = np.mean(M_alt_l, axis=0)[None, :]
            i_alt = 1 if i == 0 else 0
            M_alt = val2variants[vals[i_alt]]

            np.set_printoptions(precision=3, suppress=True)

            x_val_std = stdize(x_val, axis=1, nans=True)
            M_variants_std = stdize(M_variants, axis=1, nans=True)
            # M_variants_std = stdize(M_core[None, :], axis=1, nans=True)

            rs = np.nanmean(x_val_std * M_variants_std, axis=1)

            M_alt_std = stdize(M_alt, axis=1, nans=True)
            rs_alt = np.nanmean(x_val_std * M_alt_std, axis=1)

            df_sn.loc[df_sn[key] == val, col] = rs - rs_alt
            df_sn.loc[df_sn[key] == val, col_simple] = rs
            df_sn.loc[df_sn[key] == val, col_alt] = rs_alt
            # print(rs)
            # quit()

            # print(f'{ROI=} {val=}, {np.mean(rs - rs_alt)=:.5f}')
            # quit()

    return df_sn, MVPA_cols, MVPA_cols_simp, MVPA_cols_alt

def get_df_trialwise_MVPA(fp, key='inc', vals=(1, 3),
                          combine_regions=False):
    age2sn = get_all_sns('all', sh=False)
    sns = age2sn['healthy']
    dfs = []
    cols = []
    cols_bl = []
    cols_simp = []
    cols_alt = []
    for sn in tqdm(sns, desc='Making ERS dfs'):
        cmb_str = '_cmb' if combine_regions else ''
        fp_pkl = f'cache/smlr_MVPA_df_{sn}_{fp}_{cmb_str}.pkl'
        f = lambda: do_activity_corr_sn(sn, fp, key=key, vals=vals,
                                        combine_regions=combine_regions)
        df_sn, cols, cols_simp, cols_alt = pickle_wrap(fp_pkl, f,
                                                       easy_override=True)
        # print(f'{cols=}')
        # print(df_sn.columns)
        # quit()
        # test = df_sn.groupby(key)[cols[0]].mean()
        cols_bl = sorted(list(set([col[:-2] for col in cols])))
        # for col in cols_bl:
        #     try:
        #         df_sn[col] = (df_sn[f'{col}_L'] + df_sn[f'{col}_R']) / 2
        #     except KeyError:
        #         pass

        df_sn['sn'] = sn
        df_sn['age'] = int(sn[0])
        dfs.append(df_sn)
        # if len(dfs) > 5:
        #     break


    df = pd.concat(dfs, axis=0)
    plt.scatter(df[cols_simp[0]], df[cols_simp[0]] - df[cols_alt[0]])
    plt.xlabel('Same')
    plt.ylabel('Effect')
    plt.plot([0, 1], [0, 0], color='k')
    plt.title(f'{fp=}')
    plt.show()


    return df, cols, cols_bl, cols_alt

def do_trialwise_MVPA_analysis(fp='bl7_fMRI', key='inc', vals=(1, 3)
                               # key='inc_run', vals=(11, 13, 21, 23, 31, 33)
                               ):
    warnings.simplefilter(action='ignore',
                          category=pd.errors.PerformanceWarning)

    df, cols, cols_bl, cols_alt = get_df_trialwise_MVPA(fp,
                                                        combine_regions=True,
                                                        key=key, vals=vals)
    for col in cols:
        df_grp = df.groupby(['sn', 'inc'])[col].mean()
        # df_grp = df.groupby(['sn', 'inc'])[col].mean()
        t_inc, p = stats.ttest_rel(df_grp.loc[:, 1], df_grp.loc[:, 3],
                               nan_policy='omit')
        df_age = df.groupby(['sn', 'age'])[col].mean()
        # df_grp = df.groupby(['sn', 'inc'])[col].mean()
        try:
            t_age, p = stats.ttest_ind(df_age.loc[:, 1], df_age.loc[:, 2],
                                   nan_policy='omit')
        except KeyError:
            t_age = 0
            pass
        df_grp = df.groupby(['sn'])[col].mean()
        t0, p0 = stats.ttest_1samp(df_grp, 0, nan_policy='omit')
        M_YA = df_age.loc[:, 1].mean()
        M_OA = df_age.loc[:, 2].mean()

        print(f'{col} | Above zero: {t0=:.2f}, {p0=:.3f} | '
              f'{t_age=:.2f}, {t_inc=:.2f} [{M_YA=:.3f}, {M_OA=:.3f}]')
        # test = df.groupby(['age', 'sn', 'inc'])[col].mean().dropna()
        # print(test)
        # print('-'*10)


    quit()

def do_activity_MVPA(fp='scn7_fMRI', key='inc', vals=(1, 3),
                     combine_regions=False, age=2, corr=True):
    age2sn = get_all_sns('all', sh=False)
    sns = age2sn[age]
    dfs = []
    ROIs = []
    accs_l = []
    dfs_l = []
    for sn in tqdm(sns, desc='Making ERS dfs'):
        cmb_str = '_cmb' if combine_regions else ''
        corr_str = '_corr' if corr else ''
        fp_pkl = f'cache/ERS_df_{sn}_{fp}_{key}_{vals}{cmb_str}{corr_str}.pkl'

        accs, ROIs = pickle_wrap(fp_pkl, lambda: do_activity_MVPA_sn(sn, fp))
        accs_l.append(accs)
        df_sn = pd.DataFrame([accs], columns=ROIs, index=[sn])
        dfs_l.append(df_sn)
    accs_all = np.array(accs_l)
    ts = []
    for ROI in ROIs:
        M_acc = np.nanmean(accs_all[:, ROIs.index(ROI)])
        t, p = stats.ttest_1samp(accs_all[:, ROIs.index(ROI)], 0,
                                 nan_policy='omit')
        ts.append(t)
        if p < .05:
            print(f'{ROI=}, {M_acc=:.3f} ({t=:.2f})')
    # print('pop')
    # quit()
    atlas = get_atlas(combine_regions=combine_regions)
    fp_pic = f'mass_ttest/activity_ss-MVPA/{fp}_{key}_{age}.png'
    Path(fp_pic).parent.mkdir(parents=True, exist_ok=True)
    fp2title = {'obj7_fMRI': 'Object betas',
                'scn7_fMRI': 'Scene betas',
                'con7_fMRI': 'Conceptual retrieval betas',
                'vis7_fMRI': 'Visual retrieval betas'}
    age2str = {1: 'YA', 2: 'OA'}
    my_plot_surf(ts, atlas, f'{fp2title[fp]}, SS-MVPA\n'
                            f'{age2str[age]}, Congruent vs. Incongruent',
                 fp_out=fp_pic, neg='Bad', pos='High\nAcc.',
                 vmax=6, thresh=2.85)

def prep_data_for_group_MVPA(fp='obj7_fMRI', key='inc', vals=(1, 3),
                           combine_regions=False, age=2):
    age2sn = get_all_sns('all', sh=False)
    sns = age2sn[age]
    atlas = get_atlas(combine_regions=combine_regions)
    X_all = defaultdict(list)
    Y_all = defaultdict(list)
    groups_all = defaultdict(list)
    for sn in sns:
        df_sn = get_trial_info(sn)
        kwargs0 = {
            'sn': sn,
            'atlas': atlas,
            'fp_fMRI_col': fp,
            'df_sn': df_sn,
            'nan_thresh': 1.01,
            'drop_nan_voxels': False,
            'org_by_region': False,
            'easy_override': False,
            'combine_regions': combine_regions,
        }
        ROI2vecs0 = get_ROI_vecs(**kwargs0)
        if combine_regions:
            assert 'SFG_L' in ROI2vecs0, f'{ROI2vecs0.keys()=}'
        for ROI, vecs in ROI2vecs0.items():
            X = []
            Y = []
            for i, val in enumerate(vals):
                x_val = vecs[df_sn[key] == val, :]
                X.append(x_val)
                Y.extend([i] * x_val.shape[0])
            X = np.concatenate(X, axis=0)
            groups = [sn] * X.shape[0]
            X_all[ROI].append(X)
            Y_all[ROI].extend(Y)
            groups_all[ROI].extend(groups)
    for ROI in X_all.keys():
        X_all[ROI] = np.concatenate(X_all[ROI], axis=0)
        Y_all[ROI] = np.array(Y_all[ROI])
        groups_all[ROI] = np.array(groups_all[ROI])
    return X_all, Y_all, groups_all

def do_activity_group_MVPA(fp='obj7_fMRI', key='inc', vals=(1, 3),
                           combine_regions=False, age=2, n_repeats=10):
    cmb_str = '_cmb' if combine_regions else ''
    fp_pkl = f'cache/ERS_activity_grp_MVPA_{age}_{fp}_{key}_{vals}{cmb_str}.pkl'
    X_all, Y_all, groups_all = pickle_wrap(fp_pkl,
                               lambda: prep_data_for_group_MVPA(fp, key, vals,
                                                                combine_regions,
                                                                age),
                               easy_override=False)
    for ROI in X_all.keys():
        X = X_all[ROI]
        Y = Y_all[ROI]
        groups = groups_all[ROI]
        nan_voxels = np.isnan(X).mean(axis=0)
        good_voxels = nan_voxels <= .1
        if np.mean(good_voxels) < .25:
            print(f'{ROI=} has few ({np.mean(good_voxels)=:.1%}) good voxels')
            continue
        M = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(M, inds[1])
        X = X[:, good_voxels]
        ROI_accs = []
        for _ in tqdm(range(n_repeats), desc='group MVPA'):
            cv = GroupKFold(n_splits=2)
            clf = SVC(kernel='linear') # why is linear so slow??
            acc = cross_val_score(clf, X, Y, cv=cv, groups=groups)
            acc = np.mean(acc)
            ROI_accs.append(acc)
        ROI_acc = np.mean(ROI_accs)
        print(f'\t{ROI=} {X.shape}, {ROI_acc=:.3f}')
        continue



if __name__ == '__main__':
    # a = [10, 39, 10, 29]
    # b = [1, 2, 4, 5]
    # t, p = stats.ttest_rel(a, b)
    # print(f'{t=}')
    # quit()

    do_trialwise_MVPA_analysis()
    # analyze_ERS()
    # do_activity_MVPA_sn('102', 'obj7_fMRI')
    # do_activity_MVPA(fp='con7_fMRI', age=1)
    # do_activity_group_MVPA()
    # quit()
    # do_activity_MVPA(fp='obj7_fMRI', age=1)
    # quit()
    # do_activity_MVPA(fp='vis7_fMRI')
    # do_activity_MVPA(fp='con7_fMRI')





