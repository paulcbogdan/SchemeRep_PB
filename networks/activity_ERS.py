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
from utils import pickle_wrap, stdize
import numpy as np

from functools import wraps
from time import time
import matplotlib.pyplot as plt
import scipy.stats as stats

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

def make_ERS_df(fp0='scn7_fMRI', fp1='vis7_fMRI', combine_regions=True):
    age2sn = get_all_sns('all', sh=False)
    sns = age2sn['healthy'] # 2
    # sns = age2sn[2]
    dfs = []
    ERS_cols = []
    for sn in tqdm(sns, desc='Making ERS dfs'):
        cmb_str = '_cmb' if combine_regions else ''
        fp_pkl = f'cache/ERS_df_{sn}_{fp0}_{fp1}{cmb_str}_fixed.pkl'
        df_sn, ERS_cols = pickle_wrap(fp_pkl, lambda: do_activity_ERS_sn(sn,
                combine_regions=combine_regions, fp0=fp0, fp1=fp1,),
                                      easy_override=False)
        df_sn['age'] = int(sn[0])
        # print(df_sn['ERS_sc_vi_EVC_L'].describe())
        # quit()
        df_sn['sn'] = sn
        dfs.append(df_sn)
        # if len(dfs) > 5:
        #     break
    df = pd.concat(dfs, axis=0)
    # print(df.columns)
    # quit()
    # df.dropna(subset=['vis_hit'], inplace=True)
    for col in ERS_cols:
        df_grp = df.groupby(['sn', 'inc'])[col].mean()
        # df_grp = df.groupby(['sn', 'inc'])[col].mean()
        t_inc, p = stats.ttest_rel(df_grp.loc[:, 1], df_grp.loc[:, 3],
                               nan_policy='omit')

        df_age = df.groupby(['sn', 'age'])[col].mean()
        # df_grp = df.groupby(['sn', 'inc'])[col].mean()
        try:
            t_age, p = stats.ttest_ind(df_age.loc[:, 1], df_age.loc[:, 2],
                                   nan_policy='omit')
            df_grp = df.groupby(['sn'])[col].mean()
        except KeyError:
            t_age = 0
            pass
        t0, p0 = stats.ttest_1samp(df_grp, 0, nan_policy='omit')
        print(f'{col} | Above zero: {t0=:.2f}, {p0=:.3f} | '
              f'{t_age=:.2f}, {t_inc=:.2f}')
        continue

        if (p < .01) | (p0 < .01):
            # print(f'{col=}')
            # print(f'\t Inc effect: {t=:.2f}, {p=:.3f}')
            print(f'{col}, Above zero: {t0=:.2f}, {p0=:.3f}')
            # print(df_grp)
        continue
        formula = f'{col} ~ 1 + inc + (1 | sn)'
        keys = keys_from_formula(formula)
        model = Lmer(formula, data=df[keys].dropna())
        model.fit(REML=True, verbose=False, summary=False)
        result = model.summary()

        if (result['P-val'].iloc[0] < .01) | (result['P-val'].iloc[1] < .01):
            print(f'{col=}')
            print(result)
            print('-'*100)

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

def do_activity_MVPA(fp='obj7_fMRI', key='inc', vals=(1, 3),
                     combine_regions=False, age=2):
    age2sn = get_all_sns('all', sh=False)
    sns = age2sn[age]
    dfs = []
    ROIs = []
    accs_l = []
    dfs_l = []
    for sn in tqdm(sns, desc='Making ERS dfs'):
        cmb_str = '_cmb' if combine_regions else ''
        fp_pkl = f'cache/ERS_df_{sn}_{fp}_{key}_{vals}{cmb_str}.pkl'
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
    make_ERS_df()
    # do_activity_MVPA_sn('102', 'obj7_fMRI')
    # do_activity_MVPA(fp='obj7_fMRI')
    # do_activity_group_MVPA()
    quit()
    do_activity_MVPA(fp='obj7_fMRI', age=1)
    # quit()
    do_activity_MVPA(fp='vis7_fMRI')
    do_activity_MVPA(fp='con7_fMRI')





